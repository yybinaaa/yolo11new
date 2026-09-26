"""Validate shared inputs and portable path lists without training or inference."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import yaml

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def check(verify_hashes=False):
    raw=ROOT/'data/raw/train';fusai=ROOT/'data/fusai';split=ROOT/'data/splits/d001_train_test_4to1'
    counts=dict(raw_images=len(list(raw.glob('*.jpg'))),raw_xml=len(list(raw.glob('*.xml'))),
                fusai_images=sum(p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff') for p in fusai.rglob('*')))
    assert counts==dict(raw_images=3200,raw_xml=3200,fusai_images=788),counts
    for image in raw.glob('*.jpg'):assert image.with_suffix('.xml').is_file(),image
    with (split/'split_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    assert {s:sum(r['split']==s for r in rows) for s in ('train','test')}=={'train':2560,'test':640}
    split_hash=sha(split/'split_manifest.csv')
    for row in rows:
        for category,suffix,key in (('images',None,'image_sha256'),('labels','.txt','label_sha256')):
            p=split/category/row['split']/row['name']
            if suffix:p=p.with_suffix(suffix)
            assert p.is_file(),p
            if verify_hashes:assert sha(p)==row[key],p
    initialized=ROOT/'pretrained/yolo11m.pt'
    initialization=sha(initialized)
    weight_manifest=json.loads((ROOT/'pretrained/manifest.json').read_text(encoding='utf-8'))
    assert initialization==weight_manifest['files']['yolo11m.pt']['sha256']
    for name in ('v004_e16_v4e_split','v005_yolo11m_v4e_split'):
        version=ROOT/'versions'/name
        cfg=yaml.safe_load((version/'code/configs/experiment.yaml').read_text(encoding='utf-8'))
        assert (version/cfg['dataset']).resolve()==split.resolve()
        assert (version/cfg['pretrained']).resolve()==initialized.resolve()
        if 'expected_manifest_sha256' in cfg:assert cfg['expected_manifest_sha256']==split_hash
    lists={}
    for root in (ROOT/'data/generated').iterdir():
        if not (root/'data.yaml').exists():continue
        cfg=yaml.safe_load((root/'data.yaml').read_text(encoding='utf-8'))
        assert not cfg.get('path'),root
        for name in sorted(set(cfg[k] for k in ('train','val','test') if k in cfg)):
            listing=root/name;lines=listing.read_text(encoding='utf-8').splitlines()
            for line in lines:
                assert line.startswith('./') and '\\' not in line,(listing,line)
                image=(root/line).resolve()
                assert image.is_relative_to(root.resolve()) and image.is_file(),image
            lists[listing.relative_to(ROOT).as_posix()]=len(lines)
    return dict(status='passed',counts=counts,split=dict(train=2560,test=640,sha256=split_hash),
                initialized_sha256=initialization,portable_lists=lists,image_hashes_verified=verify_hashes,
                training_started=False,image_inference_started=False,scope='v001-v005; environment and v006 unchanged')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-hashes',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args();report=check(args.verify_hashes)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
