"""Original YOLO11m + V4E; reject custom E15/E16/DySample models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
VERSION = CODE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, nargs='+', choices=(160,170,180), default=[170])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--dataset-name', default='fusai')
    parser.add_argument('--device', default='0')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Inspect architecture without inference or training')
    args = parser.parse_args()
    if Path(args.dataset_name).name != args.dataset_name or args.dataset_name in ('.','..'):
        parser.error('dataset-name must be a single directory name')
    if not args.check_only and (args.source is None or not args.source.exists()):
        parser.error('--source must point to an existing image or directory')
    sys.path.insert(0,str(CODE))
    sys.path.insert(0,str(CODE/'scripts/runtime_bootstrap'))
    import sitecustomize  # noqa
    import _bootstrap  # noqa
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules import C3k2
    assert Path(ultralytics.__file__).resolve().is_relative_to(CODE/'ultralytics')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(CODE/'scripts/runtime_bootstrap'),str(CODE)])
    env['PYTHONIOENCODING'] = 'utf-8'
    for epoch in args.epoch:
        weights = VERSION/f'weights/epoch{epoch}.pt'
        model = YOLO(str(weights)).model
        assert type(model.model[16]) is C3k2
        assert all(type(m).__module__.startswith(('torch.','ultralytics.')) for m in model.modules())
        assert not any('ms_branch' in n for n,_ in model.named_modules())
        assert all(isinstance(model.model[i],torch.nn.Upsample) and model.model[i].mode=='nearest' for i in (11,14))
        assert model.stride.tolist()==[8.,16.,32.] and len(model.names)==9
        print(json.dumps(dict(epoch=epoch,standard_yolo11m=True,classes=model.names,
                              source=str(ultralytics.__file__)),ensure_ascii=False),flush=True)
        del model
        if args.check_only: continue
        output = (args.output_root or VERSION/'inference_packages'/args.dataset_name).resolve()/f'epoch{epoch}'
        subprocess.run([sys.executable,'-u',str(CODE/'scripts/predict_v4e.py'),
                        '--weights',str(weights),'--source',str(args.source.resolve()),
                        '--output',str(output),'--device',args.device],cwd=CODE,env=env,check=True)


if __name__ == '__main__': main()
