"""All saved local PTs -> separate, resumable competition submission packages."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

VERSION = Path(__file__).resolve().parents[1]
TOOLS = VERSION / 'inference_tools'
CODE = VERSION / 'code' if (VERSION / 'code/ultralytics').is_dir() else VERSION
NAMES = ['gunyin','huashang','jiaza','jieba','mamianmakeng','qilie','yanghuatiepi','yiwuyaru','zonglie']
SUFFIXES = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}


def source_reference(path):
    """Store shared inputs relative to the repository; allow explicit inputs on other drives."""
    path = Path(path).resolve()
    workspace = VERSION.parents[1].resolve()
    return path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else path.as_posix()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


@contextmanager
def lock(path):
    import psutil
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        try: active = abs(psutil.Process(previous['pid']).create_time() - previous['created']) < .01
        except psutil.NoSuchProcess: active = False
        if active: raise RuntimeError(f'Another inference process owns {path}')
        path.unlink()
    with path.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), f)
    try: yield
    finally: path.unlink(missing_ok=True)


def bootstrap():
    sys.path.insert(0, str(CODE))
    import runtime_support  # keeps the Windows DLL loaded
    for key, folder in [('YOLO_CONFIG_DIR','ultralytics'),('MPLCONFIGDIR','matplotlib')]:
        os.environ[key] = str(TOOLS / 'artifacts' / folder)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


def discover_weights(selected=None):
    root = VERSION / 'weights'
    if selected:
        paths = []
        for name in selected:
            if Path(name).name != name or Path(name).suffix.lower() != '.pt':
                raise ValueError('--weights accepts filenames inside this version weights/ only')
            paths.append(root / name)
    else: paths = sorted(root.glob('*.pt'))
    if len({p.name.lower() for p in paths}) != len(paths): raise ValueError('Duplicate weight names')
    for p in paths:
        if not p.is_file() or p.resolve().parent != root.resolve(): raise FileNotFoundError(p)
    return paths


def discover_images(source):
    if not source.is_dir(): raise FileNotFoundError(f'Test image directory missing: {source}')
    images = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not images: raise ValueError('No test images found')
    if len({p.name.lower() for p in images}) != len(images):
        raise ValueError('Duplicate image filenames; cannot produce unambiguous submission IDs')
    return images


def load_model(weight):
    bootstrap()
    import torch
    import yaml
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model = YOLO(str(weight))
    if [model.names[i] for i in range(len(model.names))] != NAMES:
        raise ValueError('Expected the project 9-class detector, not an official 80-class initialization')
    ckpt = model.ckpt
    epoch = int(ckpt.get('epoch', -1)) + 1
    metadata = yaml.safe_load((VERSION / 'VERSION.yaml').read_text(encoding='utf-8')) or {}
    expected = metadata.get('split_manifest_sha256')
    if expected is None and metadata.get('dataset') == 'd001_train_test_4to1':
        expected = 'a8d3acbbe9f6b42e4a0ae02fc3a93bdba4b9ad4b1964eb443e7289eb186a50e6'
    if expected and ckpt.get('split_experiment', {}).get('contract', {}).get('split') != expected:
        raise ValueError('Checkpoint does not belong to this version fixed split')
    branches = [n for n, _ in model.model.named_modules() if n.endswith('ms_branch')]
    configuration = CODE / 'configs/experiment.yaml'
    architecture = yaml.safe_load(configuration.read_text(encoding='utf-8')).get('architecture') if configuration.exists() else None
    is_e16 = architecture == 'e16' if architecture else 'e16' in VERSION.name
    if is_e16 != bool(branches): raise ValueError('Checkpoint architecture does not match version')
    return model, epoch if epoch > 0 else None


def export(items, keep_qilie):
    result = []
    for item in items:
        if item['class_name'] == 'qilie' and not keep_qilie: continue
        if item['class_name'] not in NAMES or NAMES[item['class_id']] != item['class_name']:
            raise ValueError('Invalid detection category')
        score = float(item['score']); box = item['bbox_xyxy']
        if not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4 or not all(math.isfinite(x) for x in box):
            raise ValueError('Invalid detection coordinates/score')
        box = [int(round(x)) for x in box]
        if min(box) < 0: raise ValueError('Negative detection coordinate')
        if box[2] <= box[0] or box[3] <= box[1]: continue
        result.append(dict(class_id=item['class_id'], class_name=item['class_name'], bbox_xyxy=box, score=score))
    return result


def package(out, rows, summary):
    flat = [dict(image_id=r['image_id'], category_name=d['class_name'], bbox=d['bbox_xyxy'], score=d['score'])
            for r in rows for d in r['detections']]
    write_json(out / 'predictions.json', rows)
    write_json(out / 'submission.json', flat)
    temp = out / 'submission.zip.tmp'
    with ZipFile(temp, 'w', ZIP_DEFLATED) as z: z.write(out / 'submission.json', 'submission.json')
    os.replace(temp, out / 'submission.zip')
    with ZipFile(out / 'submission.zip') as z:
        if z.namelist() != ['submission.json'] or z.testzip() is not None: raise ValueError('ZIP validation failed')
        if json.loads(z.read('submission.json')) != flat: raise ValueError('ZIP content differs from JSON')
    summary.update(status='complete', images=len(rows), detections=len(flat),
                   metrics='Not evaluated: competition test set has no ground truth in this run')
    write_json(out / 'summary.json', summary)
    write_json(out / 'checksums.json', {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()
                                      and p.name not in ('checksums.json','progress.json') and p.suffix not in ('.lock','.tmp')})


def source_digest():
    files = list(TOOLS.glob('*.py')) + [VERSION / 'INFER_FUSAI.py']
    for name in ('ultralytics','src'):
        files.extend((CODE / name).rglob('*.py'))
    return digest({p.relative_to(VERSION).as_posix():sha(p) for p in sorted(files)})


def worker(args, weight, images):
    model, epoch = load_model(weight)
    if args.check:
        print(json.dumps(dict(weight=weight.name, epoch=epoch, classes=9, architecture_checked=True), ensure_ascii=False), flush=True)
        return
    import cv2
    import numpy as np
    from v4e_core import infer
    contract = dict(version=VERSION.name, weight=weight.name, weight_sha256=sha(weight), code=source_digest(),
                    device=args.device, visualize=args.visualize, excluded_classes=[] if args.keep_qilie else ['qilie'],
                    source=source_reference(args.source), images=[dict(name=p.name, sha256=sha(p)) for p in images])
    token = digest(contract)
    out = VERSION / 'inference_packages/fusai_auto' / weight.stem / token[:16]
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / 'inference.lock'):
        cp = out / 'contract.json'
        if cp.exists() and json.loads(cp.read_text(encoding='utf-8')) != contract: raise ValueError('Output contract mismatch')
        write_json(cp, contract)
        rows = []; started = time.time()
        for index, path in enumerate(images, 1):
            cache = out / 'cache' / (path.name + '.json')
            if cache.exists():
                saved = json.loads(cache.read_text(encoding='utf-8'))
                if saved['contract'] != token or saved['row']['image_id'] != path.name: raise ValueError('Cache provenance mismatch')
                row = saved['row']
                if row['detections'] != export(row['detections'], args.keep_qilie): raise ValueError('Invalid cached detections')
            else:
                im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if im is None: raise ValueError(f'Cannot decode: {path}')
                raw = infer(im, model, args.device)
                row = dict(image_id=path.name, detections=export(raw['detections'], args.keep_qilie))
                if args.visualize:
                    for d in row['detections']:
                        x1,y1,x2,y2 = d['bbox_xyxy']
                        cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(im, f"{d['class_name']} {d['score']:.2f}",(x1,max(y1-3,15)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                    dest = out / 'visualizations' / (path.name + '.jpg'); dest.parent.mkdir(exist_ok=True)
                    ok, encoded = cv2.imencode('.jpg',im)
                    if not ok: raise RuntimeError('Could not encode visualization')
                    encoded.tofile(dest)
                write_json(cache, dict(contract=token,row=row))
            rows.append(row)
            write_json(out / 'progress.json', dict(status='running',completed=index,total=len(images)))
            print(f'{weight.name}: {index}/{len(images)} {path.name}', flush=True)
        package(out, rows, dict(weight=weight.name,epoch=epoch,elapsed_seconds=time.time()-started,excluded_classes=contract['excluded_classes']))
        write_json(out / 'progress.json',dict(status='complete',completed=len(images),total=len(images)))
    print(f'SAVED: {out}', flush=True)


def archive_outputs():
    """Extend the version index, preserving the legacy manifest schema and entries."""
    path = VERSION / 'manifest.json'
    index = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {'files':{}}
    files = index.setdefault('files', {})
    legacy = bool(files) and isinstance(next(iter(files.values())), str)
    targets = [VERSION / n for n in ('INFER_FUSAI.py','START_INFERENCE.ps1','START_INFERENCE.cmd','INFERENCE_README.md','README.md')]
    targets.extend(TOOLS.glob('*.py'))
    targets.extend((VERSION / 'inference_packages/fusai_auto').rglob('*'))
    for p in targets:
        if not p.is_file() or p.suffix in ('.lock','.tmp','.pyc') or any(x in p.relative_to(VERSION).parts for x in ('cache','visualizations')): continue
        value = sha(p)
        files[p.relative_to(VERSION).as_posix()] = value if legacy else dict(bytes=p.stat().st_size,sha256=value)
    write_json(path, index)


def main():
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=VERSION.parents[1] / 'data/fusai')
    parser.add_argument('--weights',nargs='+',help='Optional filenames from this version weights/')
    parser.add_argument('--device',default='0')
    parser.add_argument('--keep-qilie',action='store_true')
    parser.add_argument('--visualize',action='store_true')
    parser.add_argument('--check',action='store_true',help='Load and verify checkpoints, without image inference')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(); args.source = args.source.resolve()
    weights = discover_weights(args.weights)
    if not weights:
        print(f'No saved PT files in {VERSION / "weights"}. Nothing to infer.', flush=True)
        return
    images = discover_images(args.source)
    print(f'{VERSION.name}: {len(weights)} checkpoints, {len(images)} test images. Source: {args.source}', flush=True)
    if args.worker:
        if len(weights) != 1: raise ValueError('Worker requires one checkpoint')
        worker(args,weights[0],images)
        return
    output = VERSION / 'inference_packages/fusai_auto'
    with lock(TOOLS / 'artifacts/batch.lock'):
        report = dict(started_at=datetime.now().isoformat(),source=source_reference(args.source),images=len(images),results=[])
        failed = False
        try:
            for weight in weights:
                command = [sys.executable,'-u',str(VERSION/'INFER_FUSAI.py'),'--worker','--source',str(args.source),
                           '--weights',weight.name,'--device',args.device]
                for name in ('check','keep_qilie','visualize'):
                    if getattr(args,name): command.append('--'+name.replace('_','-'))
                result = subprocess.run(command, cwd=VERSION)
                report['results'].append(dict(weight=weight.name,status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode))
                failed |= result.returncode != 0
                if not args.check: write_json(output/'batch_status.json',report)
        finally:
            if not args.check:
                report['ended_at'] = datetime.now().isoformat()
                report['status'] = 'complete' if len(report['results'])==len(weights) and not failed else 'incomplete'
                write_json(output/'batch_status.json',report)
                note = '\n<!-- FUSAI_AUTO_STATUS_START -->\n最近复赛批量推理状态：'+report['status']+'；详情见 inference_packages/fusai_auto/batch_status.json。仅记录推理完成情况，未计算精度或平台成绩。\n<!-- FUSAI_AUTO_STATUS_END -->\n'
                readme = VERSION/'README.md'; text = readme.read_text(encoding='utf-8')
                start = '<!-- FUSAI_AUTO_STATUS_START -->'; end = '<!-- FUSAI_AUTO_STATUS_END -->'
                if start in text and end in text: text = text.split(start)[0].rstrip()+'\n'+text.split(end,1)[1].lstrip()
                readme.write_text(text.rstrip()+note,encoding='utf-8')
                archive_outputs()
        if failed: raise SystemExit('Some checkpoints failed. Successful results were preserved; review errors above.')
        print('CHECK PASSED (no inference).' if args.check else 'All saved checkpoints finished.',flush=True)
