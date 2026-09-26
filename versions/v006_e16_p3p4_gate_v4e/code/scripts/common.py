"""Version-local paths, full-training-data audit and run protection."""
import _bootstrap  # noqa: F401
import csv
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import psutil
import yaml

CODE = Path(__file__).resolve().parents[1]
VERSION = CODE.parent
NAMES = ['gunyin', 'huashang', 'jiaza', 'jieba', 'mamianmakeng', 'qilie', 'yanghuatiepi', 'yiwuyaru', 'zonglie']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def settings():
    c = yaml.safe_load((CODE / 'configs/experiment.yaml').read_text(encoding='utf-8'))
    if c['architecture'] != 'e16_p3p4_gate' or c['train']['val'] is not False:
        raise ValueError('V006 requires gated E16 and disabled local validation')
    if c['initialization_kind'] not in ('official', 'e16'):
        raise ValueError('initialization_kind must be official or e16')
    return c


def source_fingerprint():
    return digest({p.relative_to(CODE).as_posix(): sha256(p)
                   for sub in ('scripts', 'src', 'ultralytics', 'configs')
                   for p in sorted((CODE / sub).rglob('*'))
                   if p.is_file() and p.suffix in ('.py', '.yaml')})


def validate_training_data(c):
    """Reuse all v002 training data; never enumerate the external evaluation set."""
    path = (VERSION / c['data']).resolve()
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if data['names'] != NAMES:
        raise ValueError('Class order differs from v002')
    root = Path(data.get('path', path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    listing = root / data['train']
    lines = [s.strip() for s in listing.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if len(lines) != c['expected_train_entries'] or len(set(lines)) != len(lines):
        raise ValueError('Unexpected or duplicate training list entries')
    with (root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    full = [r for r in rows if r['sample_type'] == 'full']
    if len(full) != c['expected_source_images'] or len({r['source_image'] for r in full}) != c['expected_source_images']:
        raise ValueError('Full-data training cache does not contain all expected source images')
    if {str(Path(r['image_path']).resolve()) for r in rows} != {str((listing.parent / s).resolve()) for s in lines}:
        raise ValueError('Training list and source manifest disagree')
    stats = json.loads((root / 'statistics.json').read_text(encoding='utf-8'))
    if stats['test_images_used'] != 0:
        raise ValueError('Cache reports evaluation images used for training')
    records = []
    for index, line in enumerate(lines):
        image = (listing.parent / line).resolve()
        parts = list(image.parts)
        positions = [i for i, part in enumerate(parts) if part == 'images']
        if not positions:
            raise ValueError(f'Cannot resolve image label: {image}')
        parts[positions[-1]] = 'labels'
        label = Path(*parts).with_suffix('.txt')
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f'{image} / {label}')
        for row in label.read_text(encoding='utf-8').splitlines():
            if not row.strip():
                continue
            items = row.split()
            if len(items) != 5:
                raise ValueError(f'Invalid YOLO label: {label}')
            cls = int(items[0]); coords = list(map(float, items[1:]))
            if not 0 <= cls < 9 or not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                raise ValueError(f'Invalid YOLO label values: {label}')
        st = image.stat()
        records.append((str(image), st.st_size, st.st_mtime_ns, sha256(label)))
        if (index + 1) % 4000 == 0:
            print(f'Checked {index + 1}/{len(lines)} training entries', flush=True)
    report = dict(source_images=len(full), train_entries=len(lines), local_split_created=False,
                  evaluation_images_used=0, validation_enabled=False, yaml_sha256=sha256(path),
                  list_sha256=sha256(listing), manifest_sha256=sha256(root / 'manifest.csv'),
                  image_metadata_and_label_hash=digest(records),
                  image_bytes_hashed=False, note='Images checked for existence/size/mtime; labels checked by SHA256.')
    write_json(CODE / 'artifacts/data_check.json', report)
    return path, report


@contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        try:
            active = abs(psutil.Process(old['pid']).create_time() - old['created']) < .01
        except psutil.NoSuchProcess:
            active = False
        if active:
            raise RuntimeError(f'Another process owns {path}')
        path.unlink()
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
