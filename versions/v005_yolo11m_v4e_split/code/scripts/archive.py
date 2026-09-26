"""Refresh the release index and the README's actual run/evaluation status."""
import _bootstrap  # noqa
import json
import yaml
from common import CODE,VERSION,sha256,write_json

def refresh():
    cp=CODE/'training_records/train/checkpoint_status.json'
    completed=json.loads(cp.read_text(encoding='utf-8'))['completed_epoch'] if cp.exists() else 0
    metadata=VERSION/'VERSION.yaml'
    if metadata.exists():
        meta=yaml.safe_load(metadata.read_text(encoding='utf-8'))
        meta['trained']=completed>0;meta['completed_epoch']=completed
        meta['status']='trained' if completed else 'code_ready_not_trained'
        metadata.write_text(yaml.safe_dump(meta,sort_keys=False,allow_unicode=True),encoding='utf-8')
    entries={}
    for p in sorted(VERSION.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(VERSION);parts=rel.parts
        if '__pycache__' in parts or p.name=='manifest.json' or p.suffix in ('.pyc','.lock','.tmp'):continue
        if any(x in parts for x in ('data','cache','visualizations','artifacts')):continue
        if p.name=='README.md' and p.parent==VERSION:continue
        entries[str(rel).replace('\\','/')]=dict(bytes=p.stat().st_size,sha256=sha256(p))
    metrics=[]
    for p in sorted((VERSION/'inference_packages').glob('*/*/metrics.json')):
        m=json.loads(p.read_text(encoding='utf-8'));metrics.append((str(p.parent.relative_to(VERSION)),m))
    status=f'已完成训练轮数：{completed}。'+('尚未训练，无新训练权重。' if completed==0 else '训练记录保存在code/training_records/train。')
    status+='\n\n'
    if not metrics:status+='本地V4E尚未评测；无平台成绩。\n'
    for name,m in metrics:status+=f"- {name}：P={m['precision']:.4f}，R={m['recall']:.4f}，AP50={m['mAP50']:.4f}（本地开发验证）。\n"
    readme=VERSION/'README.md'
    if readme.exists():
        text=readme.read_text(encoding='utf-8');start='<!-- RUN_STATUS_START -->';end='<!-- RUN_STATUS_END -->'
        if start in text and end in text:text=text.split(start)[0]+start+'\n'+status+end+text.split(end,1)[1]
        readme.write_text(text,encoding='utf-8')
        entries['README.md']=dict(bytes=readme.stat().st_size,sha256=sha256(readme))
    write_json(VERSION/'manifest.json',dict(scope='Source, pretrained, trained weights, training records and inference outputs; excludes generated dataset, per-image cache, visualizations and artifacts.',files=entries))
if __name__=='__main__':refresh()
