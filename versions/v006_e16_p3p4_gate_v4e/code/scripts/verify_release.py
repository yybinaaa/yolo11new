"""Verify archived V006 hashes. No training or inference."""
import _bootstrap  # noqa: F401
import json
from common import VERSION, sha256


def main():
    manifest = json.loads((VERSION / 'manifest.json').read_text(encoding='utf-8'))
    for name, entry in manifest['files'].items():
        path = VERSION / name
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Missing or changed release file: {name}')
    print(f"Verified {len(manifest['files'])} version files")


if __name__ == '__main__': main()
