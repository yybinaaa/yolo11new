"""Audit and split original VOC images, keeping source groups and duplicates together."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack, eye
import yaml

CLASSES = ['gunyin', 'huashang', 'jiaza', 'jieba', 'mamianmakeng', 'qilie', 'yanghuatiepi', 'yiwuyaru', 'zonglie']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(path):
    annotation = path.with_suffix('.xml')
    root = ET.parse(annotation).getroot()
    width, height = int(root.findtext('size/width')), int(root.findtext('size/height'))
    with Image.open(path) as image:
        if image.size != (width, height):
            raise ValueError(f'Image/XML size mismatch: {path}')
        image.verify()
    boxes, seen, issues = [], set(), Counter()
    raw_counts = Counter()
    for obj in root.findall('object'):
        name = obj.findtext('name')
        if name not in CLASSES:
            raise ValueError(f'Unknown class {name}: {annotation}')
        raw_counts[name] += 1
        box = tuple(float(obj.findtext(f'bndbox/{key}')) for key in ['xmin', 'ymin', 'xmax', 'ymax'])
        if not np.isfinite(box).all():
            raise ValueError(f'Non-finite box: {annotation}')
        fixed = tuple(max(0, min(limit, value)) for value, limit in zip(box, (width, height, width, height)))
        issues['clipped_boxes'] += fixed != box
        if fixed[2] <= fixed[0] or fixed[3] <= fixed[1]:
            raise ValueError(f'Invalid box: {annotation}')
        key = (name, *fixed)
        if key in seen:
            issues['duplicate_boxes_removed'] += 1
            continue
        seen.add(key)
        boxes.append(key)
    return dict(name=path.name, source_group=re.split('[-_]', path.stem)[0],
                family='C_prefix' if path.name.startswith('C') else 'numeric_prefix',
                width=width, height=height, boxes=boxes, raw_counts=dict(raw_counts),
                image_sha256=sha(path), xml_sha256=sha(annotation), issues=dict(issues))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError('Output must be separate from source')
    if any((output / part).exists() for part in ['images', 'labels', 'annotations', 'manifest.json']):
        raise FileExistsError('Existing split will not be overwritten; choose a fresh output directory')
    images = sorted(source.glob('*.jpg'))
    if not images or {p.stem for p in images} != {p.stem for p in source.glob('*.xml')}:
        raise ValueError('Missing images or unmatched image/XML pairs')
    if len(images) % 5:
        raise ValueError('An exact integer 4:1 split requires a multiple of five images')
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(audit, images))
    print(f'Audited {len(records)} image/XML pairs.', flush=True)

    # Union both grouping rules, including duplicates with different filename prefixes.
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i, j):
        parents[find(j)] = find(i)
    for field in ['source_group', 'image_sha256']:
        first = {}
        for i, row in enumerate(records):
            if row[field] in first:
                union(i, first[row[field]])
            else:
                first[row[field]] = i
    grouped = defaultdict(list)
    for i in range(len(records)):
        grouped[find(i)].append(i)
    groups = sorted(grouped.values(), key=lambda indexes: records[indexes[0]]['name'])
    rng = np.random.default_rng(args.seed)
    groups = [groups[i] for i in rng.permutation(len(groups))]
    features = ['images', 'background', 'C_prefix'] + ['positive_' + c for c in CLASSES] + ['boxes_' + c for c in CLASSES]
    values = []
    for row in records:
        counts = Counter(b[0] for b in row['boxes'])
        values.append([1, int(not row['boxes']), int(row['family'] == 'C_prefix')] +
                      [int(counts[c] > 0) for c in CLASSES] + [counts[c] for c in CLASSES])
    values = np.asarray(values, dtype=float)
    matrix = np.asarray([values[indexes].sum(axis=0) for indexes in groups]).T
    target = values.sum(axis=0) / 5
    n, m = len(groups), len(features)
    # Exact image count, near-20% marginal class image counts; minimize other deviations.
    hard_lower = np.zeros(m)
    hard_upper = matrix.sum(axis=1)
    hard_lower[0] = hard_upper[0] = target[0]
    for k in range(3, 3 + len(CLASSES)):
        hard_lower[k], hard_upper[k] = max(1, np.floor(target[k]) - 1), min(matrix[k].sum() - 1, np.ceil(target[k]) + 1)
    padded = hstack([csr_matrix(matrix), csr_matrix((m, m))])
    deviations = vstack([hstack([csr_matrix(matrix), -eye(m)]), hstack([-csr_matrix(matrix), -eye(m)])])
    weights = 1 / np.maximum(target, 1)
    weights[3:3 + len(CLASSES)] *= 3
    result = milp(c=np.r_[np.zeros(n), weights], integrality=np.r_[np.ones(n), np.zeros(m)],
                  bounds=Bounds(np.zeros(n + m), np.r_[np.ones(n), np.full(m, np.inf)]),
                  constraints=[LinearConstraint(padded, hard_lower, hard_upper),
                               LinearConstraint(deviations, np.full(2 * m, -np.inf), np.r_[target, -target])],
                  options={'time_limit': 40, 'mip_rel_gap': 0.005})
    if result.x is None:
        raise RuntimeError(f'Could not find a grouped 4:1 split: {result.message}')
    selected = result.x[:n] > 0.5
    actual = matrix[:, selected].sum(axis=1)
    if not (np.all(actual >= hard_lower - 1e-6) and np.all(actual <= hard_upper + 1e-6)):
        raise RuntimeError('Solver solution failed independent constraint check')
    for group_id, indexes in enumerate(groups):
        for i in indexes:
            records[i]['group_id'] = group_id
            records[i]['split'] = 'test' if selected[group_id] else 'train'
    for field in ['name', 'source_group', 'image_sha256']:
        a = {r[field] for r in records if r['split'] == 'train'}
        b = {r[field] for r in records if r['split'] == 'test'}
        if a & b:
            raise RuntimeError(f'Cross-split leakage: {field}')
    print(f'Selected train={sum(r["split"] == "train" for r in records)}, test={int(target[0])}; copying and verifying.', flush=True)
    for kind in ['images', 'labels', 'annotations']:
        for split in ['train', 'test']:
            (output / kind / split).mkdir(parents=True)
    def export(row):
        split, stem = row['split'], Path(row['name']).stem
        image_out = output / 'images' / split / row['name']
        xml_out = output / 'annotations' / split / (stem + '.xml')
        shutil.copy2(source / row['name'], image_out)
        shutil.copy2(source / (stem + '.xml'), xml_out)
        if sha(image_out) != row['image_sha256'] or sha(xml_out) != row['xml_sha256']:
            raise RuntimeError(f'Copy checksum mismatch: {row["name"]}')
        lines = []
        for name, x1, y1, x2, y2 in row['boxes']:
            w, h = row['width'], row['height']
            lines.append(f'{CLASSES.index(name)} {(x1+x2)/(2*w):.10f} {(y1+y2)/(2*h):.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}')
        label = output / 'labels' / split / (stem + '.txt')
        label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        parsed = [line.split() for line in label.read_text().splitlines()]
        if len(parsed) != len(row['boxes']) or any(len(p) != 5 or not all(0 <= float(v) <= 1 for v in p[1:]) for p in parsed):
            raise RuntimeError(f'YOLO label verification failed: {label}')
        row['label_sha256'] = sha(label)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done, _ in enumerate(pool.map(export, records), 1):
            if done % 400 == 0:
                print(f'Copied and SHA256 verified {done}/{len(records)}', flush=True)
    summary = {}
    for split in ['all', 'train', 'test']:
        rows = [r for r in records if split == 'all' or r['split'] == split]
        summary[split] = dict(images=len(rows), background_images=sum(not r['boxes'] for r in rows),
                             positive_images=sum(bool(r['boxes']) for r in rows),
                             raw_xml_objects=sum(sum(r['raw_counts'].values()) for r in rows),
                             cleaned_yolo_objects=sum(len(r['boxes']) for r in rows),
                             source_groups=len({r['source_group'] for r in rows}),
                             families=dict(Counter(r['family'] for r in rows)))
    distribution = []
    for c in CLASSES:
        item = {'class_id': CLASSES.index(c), 'class_name': c}
        for split in ['all', 'train', 'test']:
            rows = [r for r in records if split == 'all' or r['split'] == split]
            item[split + '_images'] = sum(any(b[0] == c for b in r['boxes']) for r in rows)
            item[split + '_boxes'] = sum(sum(b[0] == c for b in r['boxes']) for r in rows)
        distribution.append(item)
    fixes = Counter()
    for row in records:
        fixes.update(row['issues'])
    duplicates = Counter(r['image_sha256'] for r in records)
    report = dict(source=str(source), seed=args.seed, requested_ratio='4:1', summary=summary,
                  sizes=dict(Counter(f'{r["width"]}x{r["height"]}' for r in records)),
                  raw_annotation_issues=dict(fixes), class_distribution=distribution,
                  groups_after_duplicate_union=len(groups), exact_duplicate_groups=sum(v > 1 for v in duplicates.values()),
                  checks=dict(image_xml_pairs=True, image_header_sizes=True, image_file_verify=True,
                              full_pixel_decode=False, exact_4_to_1=True, all_classes_in_both=True,
                              cross_split_filename_overlap=0, cross_split_source_group_overlap=0,
                              cross_split_exact_sha256_overlap=0, copied_file_hashes_verified=True,
                              yolo_labels_verified=True),
                  solver=dict(message=result.message, objective=float(result.fun), optimal=bool(result.success)),
                  limitations=['Filename prefix is a conservative source proxy, not confirmed batch metadata.',
                               'No perceptual near-duplicate audit or complete pixel decode was performed.',
                               'Existing models trained on all source images cannot yield an independent test score here.',
                               'val aliases test for loader compatibility; use only for final testing to retain holdout status.'])
    write_json(output / 'split_statistics.json', report)
    with (output / 'class_distribution.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(distribution[0]))
        writer.writeheader()
        writer.writerows(distribution)
    fields = ['name', 'split', 'source_group', 'group_id', 'image_sha256', 'xml_sha256', 'label_sha256']
    with (output / 'split_manifest.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    for split in ['train', 'test']:
        (output / f'{split}.txt').write_text(''.join(f'./images/{split}/{r["name"]}\n' for r in records if r['split'] == split), encoding='utf-8')
    (output / 'data.yaml').write_text(yaml.safe_dump(dict(train='images/train',
        val='images/test', test='images/test', names=CLASSES), allow_unicode=True, sort_keys=False), encoding='utf-8')
    write_json(output / 'source_audit.json', records)
    table = '\n'.join(f'| {d["class_name"]} | {d["all_images"]} | {d["train_images"]} | {d["test_images"]} | {d["all_boxes"]} |' for d in distribution)
    (output / 'README.md').write_text(f'''# 本地原图训练/测试划分 d001

原始数据：`{source}`。固定种子 {args.seed}。原始图片和 XML 保留不变；此目录使用独立副本，未训练、未推理、尚未评测。

## 数量

| 集合 | 图片 | 有缺陷图片 | 无缺陷图片 | 去重、边界修正后框数 |
|---|---:|---:|---:|---:|
''' + '\n'.join(f'| {s} | {v["images"]} | {v["positive_images"]} | {v["background_images"]} | {v["cleaned_yolo_objects"]} |' for s, v in summary.items()) + f'''

所有原图尺寸均为本次统计所列尺寸，详见 split_statistics.json。
原始 XML 合计 {summary['all']['raw_xml_objects']} 个框；YOLO 转换删除 {fixes['duplicate_boxes_removed']} 个完全重复框，修正 {fixes['clipped_boxes']} 个越界框，保留 {summary['all']['cleaned_yolo_objects']} 个。归档 XML 未改写。

| 类别 | 全部图片 | 训练图片 | 测试图片 | 全部有效框 |
|---|---:|---:|---:|---:|
{table}

一张图片可能包含多种缺陷，类别图片数不能直接相加作为总图片数。

## 划分与校验

按原图划分，再处理裁块。首个 `-` 或 `_` 前的文件名前缀视为同来源组；完全相同 SHA256 图片也合组，两项取传递闭包。
完整组只进入一个集合；分组优化严格保证 2560:640，同时平衡类别阳性图片数、目标框数、背景图和两种文件名前缀分布。
分组前缀仅为保守代理，未经生产批次信息确认；未做感知近重复检测。随机种子和脚本可追溯；最终复现以 split_manifest.csv 为准（限时优化器跨版本可能得到其他等价分组）。
已核对图片/XML 配对、图像头尺寸、Pillow verify、标注有效性、复制文件 SHA256、YOLO 格式、集合完整覆盖、文件名/来源组/完全重复文件跨集零重叠。没有完整解码每张图片的像素。

## 目录与使用

- `images/train/`、`images/test/`：独立图片副本。
- `annotations/train/`、`annotations/test/`：原始 VOC XML。
- `labels/train/`、`labels/test/`：与原项目类别 ID 一致的 YOLO 标签；背景图为空 TXT。
- `data.yaml`：YOLO 数据入口；`train.txt`、`test.txt`：相对当前数据目录的清单。
- `split_manifest.csv`：固定划分和逐文件 SHA256；`manifest.json`：全部交付文件校验。
- `split_statistics.json`、`class_distribution.csv`、`source_audit.json`：统计与逐图审计。
- `code/split_dataset.py`：划分脚本；依赖 numpy、scipy、Pillow、PyYAML，复用项目现有环境。

原项目默认训练配置保持原样，尚未切换至本数据集。新实验应建立单独版本，并将数据入口设为本目录 data.yaml。
若训练使用裁块或增强，只能从 images/train 创建；旧全量裁块缓存包含测试原图，不能继续混用。
历史模型若训练过全部 3200 张图，此处 640 张对它们不构成独立测试集；可靠评估需在 2560 张上重新训练，不能从已见过全部原图的项目权重续训。
只有训练/测试两份，没有另建验证集。data.yaml 的 val 与 test 均指向这 640 张，仅用于兼容加载器；若用它调参、选 epoch、早停，就应称为验证集。保留最终测试用途时应从训练部分另划验证集，或关闭训练期验证并预先固定训练方案。
最终测试可用 `model.val(data=本目录data.yaml, split="test")`；本次没有启动模型调用，也没有任何精度分数。

重新生成到新目录（从项目根目录运行；脚本拒绝覆盖已存在划分）：

```powershell
& .\\steel-defect-yolo\\.venv\\Scripts\\python.exe .\\yolo11-source-workspace\\data\\splits\\d001_train_test_4to1\\code\\split_dataset.py --source .\\yolo11-source-workspace\\data\\raw\\train --output .\\yolo11-source-workspace\\data\\splits\\d001_train_test_4to1_rebuild --seed 42
```
''', encoding='utf-8')
    files = []
    records_by_stem = {(r['split'], Path(r['name']).stem): r for r in records}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            relative = path.relative_to(output).as_posix()
            if relative.startswith(('images/', 'annotations/', 'labels/')):
                split = path.parent.name
                row = records_by_stem[(split, path.stem)]
                key = {'images': 'image_sha256', 'annotations': 'xml_sha256', 'labels': 'label_sha256'}[relative.split('/')[0]]
                digest = row[key]
            else:
                digest = sha(path)
            files.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest))
    write_json(output / 'manifest.json', dict(files=files, count=len(files), algorithm='SHA256', excludes=['manifest.json']))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
