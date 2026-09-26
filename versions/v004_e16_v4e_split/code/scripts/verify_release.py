"""Verify the recorded archive without running a model or changing files."""
import _bootstrap  # noqa
import json
from common import VERSION,sha256

def main():
    manifest=json.loads((VERSION/'manifest.json').read_text(encoding='utf-8'))
    for name,entry in manifest['files'].items():
        path=(VERSION/name).resolve()
        if not path.is_relative_to(VERSION.resolve()):raise ValueError('Invalid manifest path')
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
            raise ValueError(f'Archive mismatch: {name}')
    print(f"Verified {len(manifest['files'])} archived files; no training or inference.")
if __name__=='__main__':main()
