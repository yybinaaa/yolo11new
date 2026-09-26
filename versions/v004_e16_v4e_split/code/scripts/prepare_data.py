"""Build version-local train-only tiles plus full views; holdout stays full-image."""
import _bootstrap  # noqa
import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml
from common import CODE, PREPARED, NAMES, settings, validate_split, read_labels, sha256, digest, write_json, run_lock
from src.inference.tiler import starts, clip_box

def prep_contract(c,split):
    return dict(split=split['manifest_sha256'],options=c['preparation'],
                implementation=sha256(Path(__file__)),tiler=sha256(CODE/'src/inference/tiler.py'))

def link_or_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha256(source)!=sha256(target): raise ValueError(f'Existing file changed: {target}')
        return
    try: os.link(source,target)
    except OSError: shutil.copy2(source,target)

def labels_text(labels):
    return ''.join(f'{a[0]} '+ ' '.join(f'{v:.10f}' for v in a[1:])+'\n' for a in labels)

def plan_tiles(name,width,height,labels,opts):
    # Seed per source: resumable preparation does not change empty-tile sampling.
    rng=random.Random(f"{opts['seed']}:{name}")
    boxes=[(c,((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height)) for c,x,y,w,h in labels]
    retained=set(); planned=[]
    for y in starts(height,opts['tile_size'],opts['overlap']):
        for x in starts(width,opts['tile_size'],opts['overlap']):
            x2,y2=min(width,x+opts['tile_size']),min(height,y+opts['tile_size']); out=[]
            for i,(cls,box) in enumerate(boxes):
                cb=clip_box(box,(x,y,x2,y2),opts['min_visibility'])
                if cb and min(cb[2]-cb[0],cb[3]-cb[1])>=opts['min_box_size']:
                    a,b,c,d=cb; out.append((cls,(a+c)/2/(x2-x),(b+d)/2/(y2-y),(c-a)/(x2-x),(d-b)/(y2-y)))
                    retained.add(i)
            if not out and rng.random()>opts['keep_empty_ratio']: continue
            planned.append(dict(x=x,y=y,x2=x2,y2=y2,labels=out))
    eligible={i for i,(_,b) in enumerate(boxes) if min(b[2]-b[0],b[3]-b[1])>=opts['min_box_size']}
    if eligible-retained: raise ValueError(f'Source targets lost by tiling: {name}')
    return planned

def build(c, root, buckets, audit, output=None):
    out=Path(output or PREPARED); out.mkdir(parents=True,exist_ok=True)
    contract=prep_contract(c,audit); cp=out/'preparation_contract.json'
    if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=contract:
        raise ValueError('Preparation settings changed; use a new version/cache, never mix caches')
    if not cp.exists() and any(out.iterdir()):
        raise ValueError('Refusing an untracked existing cache')
    write_json(cp,contract)
    train=[];val=[];records=[]
    for s in ('train','test'):
        for index,r in enumerate(buckets[s]):
            n=r['name']; ip=root/'images'/s/n; lp=root/'labels'/s/Path(n).with_suffix('.txt')
            state=out/'state'/s/(n+'.json')
            if state.exists():
                entries=json.loads(state.read_text(encoding='utf-8'))
                for e in entries:
                    for key in ('image','label'):
                        if sha256(out/e[key])!=e[key+'_sha256']: raise ValueError(f'Cached {key} changed for {n}')
            else:
                entries=[]
                def record(image,label,kind):
                    entries.append(dict(source=n,source_split=s,kind=kind,image=image.relative_to(out).as_posix(),label=label.relative_to(out).as_posix(),
                                        image_sha256=sha256(image),label_sha256=sha256(label)))
                if s=='test' or c['preparation']['include_full_images']:
                    folder='val' if s=='test' else 'train'
                    full=out/'full/images'/folder/n; lab=out/'full/labels'/folder/Path(n).with_suffix('.txt')
                    link_or_copy(ip,full);link_or_copy(lp,lab);record(full,lab,'full')
                if s=='train':
                    im=cv2.imdecode(np.fromfile(ip,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                    if im is None: raise ValueError(f'Cannot decode {ip}')
                    h,w=im.shape[:2]
                    for t in plan_tiles(n,w,h,read_labels(lp),c['preparation']):
                        name=f"{Path(n).stem}__x{t['x']}_y{t['y']}.jpg"
                        image=out/'tiles/images/train'/name; label=out/'tiles/labels/train'/Path(name).with_suffix('.txt')
                        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
                        ok,encoded=cv2.imencode('.jpg',im[t['y']:t['y2'],t['x']:t['x2']])
                        if not ok: raise ValueError(f'Cannot encode {name}')
                        temp=image.with_suffix('.tmp');encoded.tofile(temp);os.replace(temp,image)
                        label.write_text(labels_text(t['labels']),encoding='utf-8');record(image,label,'tile')
                write_json(state,entries)
            records.extend(entries)
            (train if s=='train' else val).extend('./'+Path(e['image']).as_posix() for e in entries)
            if (index+1)%100==0: print(f'{s}: {index+1}/{len(buckets[s])}',flush=True)
    assert len(val)==len(buckets['test'])
    train_sources={e['source'] for e in records if e['source_split']=='train'}
    assert not train_sources&{r['name'] for r in buckets['test']}
    random.Random(c['preparation']['seed']).shuffle(train)
    for name,paths in [('train',train),('val',val)]: (out/f'{name}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
    data=dict(train='train.txt',val='val.txt',test='val.txt',names=NAMES)
    (out/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding='utf-8')
    write_json(out/'samples.json',records)
    write_json(out/'ready.json',dict(status='complete',contract=digest(contract),train_entries=len(train),val_images=len(val),
               train_sources=len(train_sources),train_holdout_overlap=0,
               files={n:sha256(out/n) for n in ('data.yaml','train.txt','val.txt','samples.json')}))
    return json.loads((out/'ready.json').read_text(encoding='utf-8'))

def check_ready(c,audit):
    out=PREPARED; p=out/'ready.json'
    if not p.exists(): raise FileNotFoundError('Run code/scripts/prepare_data.py first')
    r=json.loads(p.read_text(encoding='utf-8'))
    if r['contract']!=digest(prep_contract(c,audit)): raise ValueError('Dataset preparation contract changed')
    for n,h in r['files'].items():
        if sha256(out/n)!=h: raise ValueError(f'Prepared manifest changed: {n}')
    # Verify every label and cache file presence at each training launch.
    for e in json.loads((out/'samples.json').read_text(encoding='utf-8')):
        if not (out/e['image']).is_file() or sha256(out/e['label'])!=e['label_sha256']:
            raise ValueError(f'Prepared image missing or label changed: {e["source"]}')
    return out/'data.yaml',r

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true',help='Validate split only; no tile generation')
    p.add_argument('--verify-images',action='store_true',help='Also hash all original image bytes')
    a=p.parse_args();c=settings();root,buckets,audit=validate_split(c,a.verify_images)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    if not a.check:
        with run_lock(CODE/'artifacts/prepare.lock'): print(json.dumps(build(c,root,buckets,audit)))
if __name__=='__main__': main()
