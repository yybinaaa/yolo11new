"""Run version-local gated E16 with v002 V4E, grouped by dataset/checkpoint."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path
import re
import subprocess
import sys

from common import CODE, VERSION, run_lock, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--epoch', type=int, nargs='+', default=None)
    group.add_argument('--weights', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset-name', required=True, help='Separate name for each evaluation dataset')
    parser.add_argument('--device', default=None)
    parser.add_argument('--exclude-class', nargs='*', default=None,
                        help='Default: qilie (same as v002). Pass this option without names to retain all 9 classes.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.dataset_name):
        raise ValueError('dataset-name must contain letters, digits, underscore or hyphen')
    c = settings(); source = args.source.resolve()
    if not source.exists(): raise FileNotFoundError(source)
    weights = [args.weights.resolve()] if args.weights else [VERSION / f'weights/epoch{e}.pt' for e in (args.epoch or [180])]
    exclude = c['inference']['exclude_classes'] if args.exclude_class is None else args.exclude_class
    for weight in weights:
        if not weight.is_file(): raise FileNotFoundError(f'No trained V006 weight yet: {weight}')
        output = VERSION / 'inference_packages' / args.dataset_name / weight.stem
        try:
            with run_lock(output / 'inference.lock'):
                subprocess.run([sys.executable, '-u', str(CODE / 'scripts/v4e_core.py'),
                                '--weights', str(weight), '--source', str(source), '--output', str(output),
                                '--device', args.device or c['inference']['device'], '--exclude-class', *exclude],
                               cwd=CODE, check=True)
        finally:
            from archive import refresh
            refresh()


if __name__ == '__main__': main()
