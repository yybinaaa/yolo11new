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


import cv2
import numpy as np
def starts(length, tile, overlap):
    if tile <= 0 or length <= 0 or not 0 <= overlap < 1: raise ValueError("Invalid tile parameters")
    if length <= tile: return [0]
    step = max(1, round(tile * (1-overlap)))
    values = list(range(0, length-tile+1, step))
    if values[-1] != length-tile: values.append(length-tile)
    return values


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

