"""Read-only parent/source comparison; write version-local provenance report."""
import _bootstrap  # noqa: F401
import ast
from common import CODE, VERSION, sha256, write_json


def main():
    parent = VERSION.parent / 'v002_e16_v4e'
    hashes = {}
    for sub in ('ultralytics', 'src'):
        for source in sorted((parent / 'code' / sub).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix == '.pyc': continue
            rel = source.relative_to(parent / 'code')
            expected = sha256(source)
            if sha256(CODE / rel) != expected:
                raise ValueError(f'Inherited source changed: {rel}')
            hashes[rel.as_posix()] = expected
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(node, ast.FunctionDef) and node.name != 'main'}
    parent_core = parent / 'code/scripts/predict_e15_epoch200_v4e.py'
    if functions(parent_core) != functions(CODE / 'scripts/v4e_core.py'):
        raise ValueError('V4E algorithm functions differ from v002')
    official = VERSION.parents[2] / 'steel-defect-yolo/yolo11m.pt'
    if sha256(official) != sha256(VERSION / 'weights/initialization/yolo11m.pt'):
        raise ValueError('Official initialization copy differs')
    write_json(CODE / 'artifacts/inheritance.json', dict(parent=parent.name,
               inherited_files=hashes, inherited_file_count=len(hashes),
               v4e_algorithm_functions_equal=True, v4e_parent_sha256=sha256(parent_core),
               official_initialization_sha256=sha256(official),
               changes=['new gated E16 module', 'version-local full-data trainer and checks',
                        'dataset/checkpoint grouped inference wrapper and explicit class filter', 'archive metadata']))
    print(f'Confirmed {len(hashes)} inherited source files and unchanged V4E algorithm functions')


if __name__ == '__main__': main()
