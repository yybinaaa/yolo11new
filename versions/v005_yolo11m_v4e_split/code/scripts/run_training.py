"""Verify the frozen split, prepare version-local data, then train or resume."""
import _bootstrap  # noqa
import argparse
from datetime import datetime
import shutil
import subprocess
import sys

from common import CODE, VERSION, run_lock, settings, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Resume this version from last.pt')
    parser.add_argument('--check', action='store_true', help='Verify images and model only; no preparation or training')
    args = parser.parse_args()
    config = settings()
    runtime = CODE / 'artifacts/runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    # Reuse the local upstream AMP probe when available, avoiding a redundant download.
    probe = VERSION.parents[1] / 'pretrained/amp_probe/yolo26n.pt'
    if probe.is_file() and not (runtime / probe.name).exists():
        shutil.copy2(probe, runtime / probe.name)

    def run(script, *options):
        subprocess.run([sys.executable, '-u', str(CODE / 'scripts' / script), *options],
                       cwd=runtime, check=True)

    with run_lock(CODE / 'artifacts/launch.lock'):
        if args.check:
            run('prepare_data.py', '--check', '--verify-images')
            run('train.py', '--check')
            print('CHECK PASSED. No training or dataset tiling was started.', flush=True)
            return
        contract = CODE / 'training_records/train/contract.json'
        if args.resume:
            if not contract.is_file() or not (VERSION / 'weights/last.pt').is_file():
                raise FileNotFoundError('No complete checkpoint to resume in this version')
        elif contract.exists() or any((VERSION / 'weights').glob('*.pt')):
            raise FileExistsError('Existing run detected. Use --resume; existing records are preserved.')
        log_dir = CODE / 'training_records/launcher'
        log_dir.mkdir(parents=True, exist_ok=True)
        status = dict(started_at=datetime.now().isoformat(), resume=args.resume,
                      epochs=config['train']['epochs'], save_epochs=config['checkpoints']['save_epochs'],
                      split_sha256=config['expected_manifest_sha256'], status='preparing')
        status_path = log_dir / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
        write_json(status_path, status)
        try:
            run('prepare_data.py', '--verify-images')
            status['status'] = 'training'
            write_json(status_path, status)
            run('train.py', *(['--resume'] if args.resume else []))
            status['status'] = 'finished'
        except BaseException as exc:
            status['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            status['error'] = str(exc)
            raise
        finally:
            status['ended_at'] = datetime.now().isoformat()
            write_json(status_path, status)
            from archive import refresh
            refresh()


if __name__ == '__main__':
    main()
