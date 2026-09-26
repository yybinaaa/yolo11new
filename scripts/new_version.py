"""Create an untrained version, sharing workspace data and initialization weights."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import yaml
from common import DEFAULT_CONFIG, WORKSPACE_ROOT

VERSION_PATTERN = re.compile(r"^v\d{3}_[a-z0-9][a-z0-9_]*$")

def create_version(name, source_name, description, config_path=DEFAULT_CONFIG):
    for label in (name, source_name):
        if not VERSION_PATTERN.fullmatch(label):
            raise ValueError("版本名必须类似 v007_yolo11m_example")
    source = WORKSPACE_ROOT / 'versions' / source_name
    destination = WORKSPACE_ROOT / 'versions' / name
    if not source.is_dir(): raise FileNotFoundError(source)
    if destination.exists(): raise FileExistsError(f'拒绝覆盖已有版本: {destination}')
    split_style = (source / 'code/configs/experiment.yaml').exists()
    raw = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    if split_style:
        cfg = yaml.safe_load((source/'code/configs/experiment.yaml').read_text(encoding='utf-8'))
        if 'dataset' not in cfg:
            raise ValueError('此创建入口目前支持基线和 v004/v005 的划分训练结构；v006 尚未迁移。')
    elif source_name not in raw.get('versions', {}):
        raise ValueError('该历史快照没有独立训练配置，请选择 v001、v004 或 v005 为父版本。')

    def ignore(directory, names):
        relative = Path(directory).relative_to(source)
        excluded = {n for n in names if n in ('__pycache__', '.pytest_cache') or n.endswith('.pyc')}
        if relative == Path('.'):
            allowed = {'code','models','ultralytics','inference_tools','INFER_FUSAI.py','LICENSE',
                       'START_TRAINING.ps1','START_TRAINING.cmd','START_INFERENCE.ps1','START_INFERENCE.cmd'}
            excluded.update(set(names)-allowed)
        if relative == Path('code'):
            excluded.update(set(names)&{'data','artifacts','training_records','training_archive','pretrained','outputs','runs'})
        if relative == Path('inference_tools'):
            excluded.update(set(names)&{'artifacts'})
        return excluded

    shutil.copytree(source, destination, ignore=ignore)
    for folder in ('weights','inference_packages'):
        (destination/folder).mkdir(exist_ok=True)
    if split_style:
        cfg['dataset']='../../data/splits/d001_train_test_4to1'
        cfg['pretrained']='../../pretrained/yolo11m.pt'
        (destination/'code/configs/experiment.yaml').write_text(
            yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
        commands = 'python code/scripts/prepare_data.py --check --verify-images\npython code/scripts/train.py --check\n# 以下命令会准备数据并启动训练，仅在需要时执行\npython code/scripts/prepare_data.py --verify-images\npython code/scripts/train.py'
    else:
        raw['versions'][name] = copy.deepcopy(raw['versions'][source_name])
        raw['versions'][name].update(description=description,parent=source_name)
        config_path.write_text(yaml.safe_dump(raw,sort_keys=False,allow_unicode=True),encoding='utf-8')
        commands = f'python ../../scripts/train.py --version {name} --dry-run\n# 以下命令启动训练\npython ../../scripts/train.py --version {name}'
    meta=dict(name=name,parent=source_name,description=description,
              created_at=datetime.now().isoformat(timespec='seconds'),status='code_ready_not_trained',
              trained=False,completed_epoch=0,license='AGPL-3.0',shared_layout=1)
    if split_style:
        meta.update(architecture=cfg['architecture'],dataset='d001_train_test_4to1')
        if 'expected_manifest_sha256' in cfg:meta['split_manifest_sha256']=cfg['expected_manifest_sha256']
    (destination/'VERSION.yaml').write_text(yaml.safe_dump(meta,sort_keys=False,allow_unicode=True),encoding='utf-8')
    (destination/'README.md').write_text(
        f'# {name}\n\n{description}\n\n父版本：{source_name}。只复制源码与运行配置，未复制父版本训练权重、缓存或结果。\n\n'
        f'共享数据：`../../data/`；初始化权重：`../../pretrained/yolo11m.pt`。\n'+
        ('训练结果仍保存在本版本 `weights/`、`code/training_records/` 和 `inference_packages/`。\n\n' if split_style else
         f'基线沿用共享切片 data/generated/v001_baseline/；训练输出为仓库 runs/{name}/，配置见根目录 configs/experiment.yaml。\n\n')+
        '以下命令在本版本目录、已安装依赖的环境中运行：\n\n```text\n'+commands+'\n```\n\n'
        +('训练产生本版本权重后，可用 `python INFER_FUSAI.py --check --weights best.pt` 只检查加载；'
          '`python INFER_FUSAI.py --weights best.pt` 会对共享复赛图片执行真实推理。\n\n' if split_style else '')+
        '<!-- RUN_STATUS_START -->\n尚未训练、尚未评测，无精度成绩。\n<!-- RUN_STATUS_END -->\n',encoding='utf-8')
    entries={}
    for p in sorted(destination.rglob('*')):
        if p.is_file(): entries[p.relative_to(destination).as_posix()]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (destination/'manifest.json').write_text(json.dumps(dict(files=entries),ensure_ascii=False,indent=2),encoding='utf-8')
    return destination

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name');p.add_argument('--from-version',default='v005_yolo11m_v4e_split',dest='source')
    p.add_argument('--description',required=True);p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    a=p.parse_args()
    print(create_version(a.name,a.source,a.description,a.config.resolve()))
    print('仅创建源码版本；没有准备数据、训练或推理。')

if __name__=='__main__':main()
