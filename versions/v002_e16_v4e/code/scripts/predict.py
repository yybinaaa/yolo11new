"""Run the archived E16 checkpoints with version-local Ultralytics and V4E."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
VERSION = CODE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, nargs='+', choices=(160,170,180), default=[180])
    parser.add_argument('--source', type=Path, required=True, help='Image directory or single image')
    parser.add_argument('--output-root', type=Path, default=VERSION/'outputs')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.exists(): raise FileNotFoundError(source)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(CODE/'scripts/runtime_bootstrap'),str(CODE)])
    env['PYTHONIOENCODING'] = 'utf-8'
    for epoch in args.epoch:
        subprocess.run([sys.executable, '-u', str(CODE/'scripts/predict_e15_epoch200_v4e.py'),
                        '--weights',str(VERSION/f'weights/epoch{epoch}.pt'),
                        '--source',str(source),'--output',str(args.output_root.resolve()/f'epoch{epoch}'),
                        '--device',args.device], cwd=CODE,env=env,check=True)


if __name__ == '__main__': main()
