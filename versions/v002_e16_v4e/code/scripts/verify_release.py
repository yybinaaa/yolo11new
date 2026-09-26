"""Verify release files against the packaged SHA256 manifest (no training)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = ROOT/relative
        actual = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if actual != expected: raise ValueError(f'Hash mismatch: {relative}')
    print(f"Verified {len(manifest['files'])} release files")


if __name__ == '__main__': main()
