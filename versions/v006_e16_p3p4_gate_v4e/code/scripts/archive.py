"""Update V006 status and hashes without running training or inference."""
import _bootstrap  # noqa: F401
import json
import yaml
from common import CODE, VERSION, sha256, write_json


def refresh():
    status_file = CODE / 'training_records/train/checkpoint_status.json'
    completed = json.loads(status_file.read_text(encoding='utf-8'))['completed_epoch'] if status_file.exists() else 0
    summaries = []
    for path in sorted((VERSION / 'inference_packages').glob('*/*/inference_summary.json')):
        summary = json.loads(path.read_text(encoding='utf-8'))
        summaries.append(dict(directory=path.parent.relative_to(VERSION).as_posix(),
                              status=summary['status'], images=summary['images'], detections=summary['detections']))
    metadata = VERSION / 'VERSION.yaml'
    meta = yaml.safe_load(metadata.read_text(encoding='utf-8'))
    meta.update(trained=completed > 0, completed_epoch=completed,
                status='trained_not_platform_evaluated' if completed else 'code_ready_not_trained')
    metadata.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding='utf-8')
    readme = VERSION / 'README.md'
    content = readme.read_text(encoding='utf-8')
    start, end = '<!-- RUN_STATUS_START -->', '<!-- RUN_STATUS_END -->'
    status = f'已完成训练轮数：{completed}。'
    status += '尚未训练；weights/initialization/仅包含初始化权重。' if completed == 0 else '训练PT位于weights/，记录位于code/training_records/train/。'
    status += '\n\n尚无本版本平台评测成绩；不以训练损失、预测框数或结构检查代替精度。\n'
    if not summaries: status += '\n尚未执行真实评测集V4E推理。\n'
    for item in summaries:
        status += f"\n- {item['directory']}：{item['images']}张图，{item['detections']}个预测框；框数不是精度。\n"
    if start not in content or end not in content: raise ValueError('README status markers missing')
    readme.write_text(content.split(start)[0] + start + '\n' + status + end + content.split(end, 1)[1], encoding='utf-8')
    files = {}
    for path in sorted(VERSION.rglob('*')):
        if not path.is_file(): continue
        rel = path.relative_to(VERSION)
        if '__pycache__' in rel.parts or path.suffix in ('.pyc', '.tmp', '.lock'): continue
        if rel.as_posix() == 'manifest.json': continue
        if rel.parts[:3] in (('code', 'artifacts', 'ultralytics'), ('code', 'artifacts', 'matplotlib')): continue
        files[rel.as_posix()] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    write_json(VERSION / 'manifest.json', dict(scope='All version files, including checks and inference results; excludes runtime caches, pyc, locks and temporary files.', files=files))
    print(f'Archive refreshed: {len(files)} files, {completed} trained epochs')


if __name__ == '__main__': refresh()
