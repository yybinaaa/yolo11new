"""Independent implementation of the user supplied SAHI V4E pseudocode.

Clustering uses same-class connected components (IoU OR IoM). Region merging
is greedy against the highest-priority cluster center, with Euclidean distance.
Internal border distance uses box edges, excluding the outer image boundary.
Model-local NMS uses IoU=.7; cross-view NMS uses .5. No class threshold YAML.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import _bootstrap  # noqa
import cv2
import numpy as np
from predict_epoch93_sahi import ROOT, DEFAULT_SOURCE, list_images, annotate
from build_submission import convert_predictions, save_submission
from src.inference.tiler import starts


def overlap(a, b):
    inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    aa = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return inter/max(aa+bb-inter, 1e-12), inter/max(min(aa, bb), 1e-12)


def center(d):
    b = d['bbox_xyxy']
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)


def origin(c, size, width, height):
    return (max(0, min(round(c[0]-size/2), max(0, width-size))),
            max(0, min(round(c[1]-size/2), max(0, height-size))))


def nms(items, threshold=.5):
    kept = []
    for d in sorted(items, key=lambda d: -d['score']):
        if not any(d['class_id'] == k['class_id'] and overlap(d['bbox_xyxy'], k['bbox_xyxy'])[0] >= threshold for k in kept):
            kept.append(d)
    return kept


def border_distance(d, width, height):
    if d['source'] != 'tile':
        return math.inf
    x, y = d['tile_origin']
    tw, th = d['view_size']
    x1, y1, x2, y2 = d['bbox_xyxy']
    distances = []
    if x > 0: distances.append(abs(x1-x))
    if y > 0: distances.append(abs(y1-y))
    if x+tw < width: distances.append(abs(x+tw-x2))
    if y+th < height: distances.append(abs(y+th-y2))
    return min(distances, default=math.inf)


def adaptive_origins(first, width, height):
    triggers = []
    for d in first:
        s = d['score']
        border = .10 <= s < .50 and border_distance(d, width, height) <= 96
        if .08 <= s < .25 or border:
            triggers.append((origin(center(d), 1024, width, height), s+(2 if border else 1)))
    kept = []
    for xy, _ in sorted(triggers, key=lambda t: -t[1]):
        if all(math.dist(xy, p) >= 192 for p in kept):
            kept.append(xy)
        if len(kept) == 4: break
    return kept


def weak_regions(first, anchors):
    weak = [d for d in first if .05 <= d['score'] < .18 and not any(
        d['class_id'] == a['class_id'] and overlap(d['bbox_xyxy'], a['bbox_xyxy'])[0] >= .30 for a in anchors)]
    parents = list(range(len(weak)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, a in enumerate(weak):
        for j in range(i):
            b = weak[j]
            if a['class_id'] == b['class_id']:
                iou, iom = overlap(a['bbox_xyxy'], b['bbox_xyxy'])
                if iou >= .25 or iom >= .55: parents[root(i)] = root(j)
    groups = {}
    for i, d in enumerate(weak): groups.setdefault(root(i), []).append(d)
    clusters = []
    for seeds in groups.values():
        support = len({d['view_id'] for d in seeds})
        # Score-weighted mean of seed centers defines the cluster location.
        c = np.average([center(d) for d in seeds], axis=0, weights=[d['score'] for d in seeds]).tolist()
        clusters.append(dict(center=c, seeds=seeds, support=support,
                             priority=max(d['score'] for d in seeds)+.025*min(3, support-1)))
    regions = []
    for cluster in sorted(clusters, key=lambda c: -c['priority']):
        match = next((r for r in regions if math.dist(r['center'], cluster['center']) < 192), None)
        if match is None: regions.append(cluster)
        else: match['seeds'].extend(cluster['seeds'])
    return regions[:2]


def safe_add(anchors, candidates):
    output = list(anchors)
    for d in nms(candidates):
        if not any(d['class_id'] == a['class_id'] and overlap(d['bbox_xyxy'], a['bbox_xyxy'])[0] >= .5 for a in output):
            output.append(d)
    assert output[:len(anchors)] == anchors
    return output


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def infer(image, model, device):
    height, width = image.shape[:2]
    def predict(x, y, size, source):
        roi = image if source == 'full' else image[y:y+size, x:x+size]
        result = model.predict(roi, imgsz=1024, conf=.05, iou=.7, max_det=1000,
                               device=device, verbose=False)[0]
        found = []
        for row in result.boxes.data.cpu().tolist():
            x1, y1, x2, y2, score, cls = row
            box = [max(0., min(width, x1+x)), max(0., min(height, y1+y)),
                   max(0., min(width, x2+x)), max(0., min(height, y2+y))]
            if box[2] <= box[0] or box[3] <= box[1]: continue
            found.append(dict(bbox_xyxy=box, score=score, class_id=int(cls),
                              class_name=str(model.names[int(cls)]), source=source,
                              tile_origin=[x,y], view_size=[roi.shape[1],roi.shape[0]],
                              view_id=f'{source}:{x}:{y}:{size}'))
        return found
    first = []
    for y in starts(height, 1024, .20):
        for x in starts(width, 1024, .20): first.extend(predict(x,y,1024,'tile'))
    first.extend(predict(0,0,1024,'full'))
    origins = adaptive_origins(first,width,height)
    refined = []
    for x,y in origins: refined.extend(predict(x,y,1024,'adaptive'))
    anchors = nms([d for d in first+refined if d['score'] >= .18])
    regions = weak_regions(first,anchors)
    confirmations = []
    for region in regions:
        for size in (640,1280):
            x,y = origin(region['center'],size,width,height)
            for d in predict(x,y,size,f'v4_{size}'):
                if d['score'] < .10: continue
                if any(d['class_id'] == s['class_id'] and (
                    overlap(d['bbox_xyxy'],s['bbox_xyxy'])[0] >= .10 or
                    overlap(d['bbox_xyxy'],s['bbox_xyxy'])[1] >= .50) for s in region['seeds']):
                    confirmations.append(d)
    output = safe_add(anchors,confirmations)
    return dict(first_pass=first, adaptive_origins=origins, refinement=refined,
                anchors=anchors, weak_regions=regions, v4_confirmations=confirmations,
                detections=output, added=len(output)-len(anchors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=ROOT.parent/'weights/epoch180.pt')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'inference_packages/fusai/v4e_manual_epoch180')
    parser.add_argument('--device',default='0')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    files = list_images(args.source)
    contract = dict(weights=str(args.weights.resolve()), weight_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    images=[dict(name=p.name,size=p.stat().st_size,mtime=p.stat().st_mtime_ns) for p in files])
    contract_path = args.output/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text(encoding='utf-8')) != contract:
        raise ValueError('Output contract differs; use a new output directory')
    write_json(contract_path,contract)
    cache = args.output/'cache'
    cache.mkdir(exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    records, anchor_records = [], []
    started = time.time()
    try:
        for index,path in enumerate(files):
            cached = cache/(path.name+'.json')
            if cached.exists(): data = json.loads(cached.read_text(encoding='utf-8'))
            else:
                image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is None: raise ValueError(f'Cannot read {path}')
                data = infer(image,model,args.device)
                # Filter only after all V4E stages, matching the submitted exporter.
                for key in ('anchors', 'detections'):
                    data[key] = [d for d in data[key] if d['class_name'] != 'qilie']
                data['added'] = len(data['detections'])-len(data['anchors'])
                assert data['detections'][:len(data['anchors'])] == data['anchors']
                ok,encoded = cv2.imencode('.jpg',annotate(image,data['detections']))
                if not ok: raise ValueError(f'Cannot encode {path}')
                encoded.tofile(args.output/(path.stem+'.jpg'))
                write_json(cached,data)
            records.append(dict(image_id=path.name,detections=data['detections']))
            anchor_records.append(dict(image_id=path.name,detections=data['anchors']))
            write_json(args.output/'progress.json',dict(status='running',completed=index+1,total=len(files),
                       detections=sum(len(r['detections']) for r in records),elapsed_seconds=time.time()-started))
            print(f'{index+1}/{len(files)} {path.name}: anchors={len(data["anchors"])} added={data["added"]}',flush=True)
        for name, rows in [('v4e',records),('v3e_anchor',anchor_records)]:
            write_json(args.output/f'{name}_predictions.json',rows)
            flat = convert_predictions(rows,args.source if args.source.is_dir() else None)
            save_submission(flat,args.output/f'{name}_submission.json',args.output/f'{name}_submission.zip')
        anchor_count = sum(len(r['detections']) for r in anchor_records)
        total = sum(len(r['detections']) for r in records)
        write_json(args.output/'inference_summary.json',dict(status='complete',images=len(files),
                   anchors=anchor_count,added=total-anchor_count,detections=total,
                   anchor_preservation_verified=True,elapsed_seconds=time.time()-started,
                   weights=str(args.weights),excluded_categories=['qilie'],metrics='No ground truth supplied; submit ZIP for scoring'))
        write_json(args.output/'progress.json',dict(status='complete',completed=len(files),total=len(files)))
    except BaseException as exc:
        write_json(args.output/'error.json',dict(error=repr(exc),completed=len(records)))
        raise


if __name__ == '__main__': main()
