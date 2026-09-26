"""V4E inference. Defaults to the 640-image local holdout and keeps all 9 classes."""
import _bootstrap  # noqa
import argparse
import json
import re
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import cv2
import numpy as np
from common import CODE, VERSION, NAMES, settings, validate_split, sha256, write_json, source_fingerprint, run_lock
from v4e_core import infer

def exported(items,exclude):
    result=[]
    for d in items:
        if d['class_name'] in exclude: continue
        b=[int(round(x)) for x in d['bbox_xyxy']]
        if b[2]<=b[0] or b[3]<=b[1]: continue
        result.append(dict(class_name=d['class_name'],class_id=d['class_id'],bbox_xyxy=b,score=d['score']))
    return result

def save_submission(rows,dest):
    flat=[dict(image_id=r['image_id'],category_name=d['class_name'],bbox=d['bbox_xyxy'],score=d['score']) for r in rows for d in r['detections']]
    write_json(dest.with_suffix('.json'),flat)
    with ZipFile(dest.with_suffix('.zip'),'w',ZIP_DEFLATED) as z:z.write(dest.with_suffix('.json'),arcname='submission.json')

def run(args):
    from ultralytics import YOLO
    from train import architecture_check
    c=settings();root,buckets,audit=validate_split(c)
    weights=(Path(args.weights).resolve() if args.weights else VERSION/'weights'/('best.pt' if args.epoch is None else f'epoch{args.epoch}.pt'))
    if not weights.is_file(): raise FileNotFoundError(f'Train this version first. Missing: {weights}')
    model=YOLO(str(weights));architecture_check(model.model,c['architecture'])
    meta=model.ckpt.get('split_experiment',{})
    if meta.get('architecture')!=c['architecture'] or meta.get('contract',{}).get('split')!=audit['manifest_sha256']:
        raise ValueError('Weights lack matching new-split training provenance; historical full-data models are not allowed')
    if list(model.names.values())!=NAMES: raise ValueError('Weight class mapping mismatch')
    if args.source:
        if not args.dataset_name: raise ValueError('--source requires --dataset-name')
        source=Path(args.source).resolve();files=sorted([source] if source.is_file() else [p for p in source.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')])
        dataset=args.dataset_name
    else:
        source=root/'images/test';files=[source/r['name'] for r in buckets['test']];dataset=args.dataset_name or c['inference']['default_dataset_name']
    if not files: raise ValueError('No images')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',dataset): raise ValueError('Use a simple dataset name')
    if len({p.name for p in files})!=len(files): raise ValueError('Duplicate filenames')
    exclude=c['inference']['exclude_classes'] if args.exclude_class is None else args.exclude_class
    if any(n not in NAMES for n in exclude):raise ValueError('Unknown excluded class')
    if set(exclude)==set(NAMES):raise ValueError('Cannot exclude every class')
    epoch=int(model.ckpt['epoch'])+1
    out=VERSION/'inference_packages'/dataset/f'epoch{epoch}'
    contract=dict(weights_sha256=sha256(weights),split=audit['manifest_sha256'],code=source_fingerprint(),
                  exclude_classes=sorted(exclude),device=args.device,visualize=args.visualize,
                  images=[dict(name=p.name,sha256=sha256(p)) for p in files])
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out/'inference.lock'):
        cp=out/'contract.json'
        if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=contract:
            raise ValueError('Output contract changed. Use a different --dataset-name.')
        if not cp.exists() and any(p.name!='inference.lock' for p in out.iterdir()): raise ValueError('Untracked output directory')
        write_json(cp,contract);records=[];anchors=[];start=time.time()
        for i,path in enumerate(files):
            cached=out/'cache'/(path.name+'.json')
            if cached.exists():data=json.loads(cached.read_text(encoding='utf-8'))
            else:
                image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is None:raise ValueError(f'Cannot decode {path}')
                raw=infer(image,model,args.device)
                data=dict(detections=exported(raw['detections'],exclude),anchors=exported(raw['anchors'],exclude))
                if data['detections'][:len(data['anchors'])]!=data['anchors']:raise RuntimeError('Anchor preservation failed')
                if args.visualize:
                    for d in data['detections']:
                        x1,y1,x2,y2=d['bbox_xyxy'];cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(image,f"{d['class_name']} {d['score']:.2f}",(x1,max(15,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                    vis=out/'visualizations';vis.mkdir(exist_ok=True);ok,jpg=cv2.imencode('.jpg',image)
                    if not ok:raise RuntimeError('Visualization encoding failed')
                    jpg.tofile(vis/(path.stem+'.jpg'))
                write_json(cached,data)
            records.append(dict(image_id=path.name,detections=data['detections']));anchors.append(dict(image_id=path.name,detections=data['anchors']))
            write_json(out/'progress.json',dict(completed=i+1,total=len(files),status='running'))
            print(f'{i+1}/{len(files)} {path.name}: {len(data["detections"])}',flush=True)
        for name,rows in [('v4e',records),('v3e_anchor',anchors)]:
            write_json(out/f'{name}_predictions.json',rows);save_submission(rows,out/f'{name}_submission')
        write_json(out/'inference_summary.json',dict(status='complete',images=len(files),epoch=epoch,architecture=c['architecture'],
                    detections=sum(len(r['detections']) for r in records),anchors=sum(len(r['detections']) for r in anchors),
                    excluded_classes=exclude,elapsed_seconds=time.time()-start,metrics='Not evaluated; run evaluate.py for local holdout'))
        write_json(out/'progress.json',dict(completed=len(files),total=len(files),status='complete'))
    from archive import refresh
    refresh()
    return out

def arguments():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--weights');g.add_argument('--epoch',type=int)
    p.add_argument('--source');p.add_argument('--dataset-name');p.add_argument('--device',default=settings()['inference']['device'])
    p.add_argument('--exclude-class',nargs='*',default=None);p.add_argument('--visualize',action='store_true')
    return p.parse_args()
if __name__=='__main__':run(arguments())
