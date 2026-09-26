"""Version-local paths, split validation, atomic records and run protection."""
import _bootstrap  # noqa
import csv
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import psutil
import yaml

CODE = Path(__file__).resolve().parents[1]
VERSION = CODE.parent
WORKSPACE = VERSION.parents[1]
PREPARED = WORKSPACE / "data/generated" / VERSION.name
NAMES = ['gunyin','huashang','jiaza','jieba','mamianmakeng','qilie','yanghuatiepi','yiwuyaru','zonglie']

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)

def settings():
    c=yaml.safe_load((CODE/'configs/experiment.yaml').read_text(encoding='utf-8'))
    if c['architecture'] not in ('e16','baseline'): raise ValueError('Unknown architecture')
    return c

def dataset_root(c): return (VERSION/c['dataset']).resolve()

def validate_split(c, verify_images=False):
    root=dataset_root(c)
    with (root/'split_manifest.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    data=yaml.safe_load((root/'data.yaml').read_text(encoding='utf-8'))
    if data['names'] != NAMES: raise ValueError('Class mapping changed')
    seen=set(); buckets={s:[] for s in ('train','test')}; keys={s:{k:set() for k in ('group_id','source_group','image_sha256')} for s in buckets}
    for r in rows:
        s=r['split']; n=r['name']
        if s not in buckets or Path(n).name!=n or n in seen: raise ValueError('Invalid/duplicate split record')
        seen.add(n); buckets[s].append(r)
        for key in keys[s]: keys[s][key].add(r[key])
        im=root/'images'/s/n; lb=root/'labels'/s/Path(n).with_suffix('.txt')
        if not im.is_file() or not lb.is_file(): raise FileNotFoundError(n)
        if sha256(lb)!=r['label_sha256']: raise ValueError(f'Label hash changed: {lb}')
        if verify_images and sha256(im)!=r['image_sha256']: raise ValueError(f'Image hash changed: {im}')
        read_labels(lb)
    for key in keys['train']:
        if keys['train'][key]&keys['test'][key]: raise ValueError(f'Train/test leakage: {key}')
    actual={s:len(v) for s,v in buckets.items()}
    if actual != c['expected_split']: raise ValueError(f'Unexpected split counts: {actual}')
    for s in buckets:
        present={p.name for p in (root/'images'/s).iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')}
        if present!={r['name'] for r in buckets[s]}: raise ValueError(f'Untracked images in {s}')
    return root,buckets,dict(counts=actual,manifest_sha256=sha256(root/'split_manifest.csv'),
                             labels_verified=True,image_bytes_verified=verify_images,overlap=0)

def read_labels(path):
    import math
    result=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        parts=line.split()
        if len(parts)!=5: raise ValueError(f'Invalid label: {path}')
        cls=int(parts[0]); v=[float(x) for x in parts[1:]]
        if not 0<=cls<len(NAMES) or not all(math.isfinite(x) and 0<=x<=1 for x in v) or min(v[2:])<=0:
            raise ValueError(f'Invalid label: {path}')
        result.append((cls,*v))
    return result

def source_fingerprint():
    return digest({p.relative_to(CODE).as_posix():sha256(p) for sub in ('scripts','src','ultralytics','configs')
                   for p in sorted((CODE/sub).rglob('*')) if p.is_file() and p.suffix in ('.py','.yaml')})

@contextmanager
def run_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        try: active=abs(psutil.Process(old['pid']).create_time()-old['created'])<.01
        except psutil.NoSuchProcess: active=False
        if active: raise RuntimeError(f'Another process owns {path}')
        path.unlink()
    token=dict(pid=os.getpid(),created=psutil.Process().create_time())
    with path.open('x',encoding='utf-8') as f: json.dump(token,f)
    try: yield
    finally: path.unlink(missing_ok=True)

