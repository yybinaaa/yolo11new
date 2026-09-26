"""Explicit local class-wise one-to-one IoU evaluation on ALL holdout images."""
from collections import defaultdict
import math
import numpy as np
from PIL import Image
from common import NAMES, read_labels

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-12)

def load_gt(root,rows):
    result={}
    for r in rows:
        name=r['name']
        with Image.open(root/'images/test'/name) as image: w,h=image.size
        result[name]=[dict(class_name=NAMES[c],bbox_xyxy=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
                      for c,x,y,bw,bh in read_labels(root/'labels/test'/__import__('pathlib').Path(name).with_suffix('.txt'))]
    return result

def evaluate(predictions,truth,threshold=.5,exclude=()):
    names=[n for n in NAMES if n not in exclude]
    if threshold != .5: raise ValueError('This AP50 evaluator requires matching IoU=0.50')
    byimage={}
    for row in predictions:
        n=row['image_id']
        if n not in truth or n in byimage: raise ValueError('Unknown/duplicate image in predictions')
        byimage[n]=row['detections']
        for d in row['detections']:
            b=d['bbox_xyxy'];s=d['score']
            if d['class_name'] not in NAMES or len(b)!=4 or not all(math.isfinite(x) for x in b) or b[2]<=b[0] or b[3]<=b[1] or not math.isfinite(s) or not 0<=s<=1:
                raise ValueError('Invalid detection')
    if set(byimage)!=set(truth): raise ValueError('Predictions must include every holdout image, including zero-detection images')
    report={};all_tp=all_fp=all_gt=0
    size_counts={k:dict(tp=0,gt=0) for k in ('short_lt16','short_16_to32','short_ge32','long_gt1024')}
    for cls in names:
        gt={im:[d['bbox_xyxy'] for d in ds if d['class_name']==cls] for im,ds in truth.items()}
        used={im:set() for im in gt};ranked=sorted([(im,d) for im,ds in byimage.items() for d in ds if d['class_name']==cls],key=lambda x:-x[1]['score'])
        ng=sum(map(len,gt.values()));tp=fp=0;rec=[];pre=[]
        for im,d in ranked:
            candidates=[(iou(d['bbox_xyxy'],b),j) for j,b in enumerate(gt[im]) if j not in used[im]]
            score,j=max(candidates,default=(-1,-1))
            if score>=threshold: tp+=1;used[im].add(j)
            else: fp+=1
            rec.append(tp/ng if ng else 0);pre.append(tp/(tp+fp))
        ap=float(np.mean([max((p for r,p in zip(rec,pre) if r>=q),default=0) for q in np.linspace(0,1,101)])) if ng else None
        report[cls]=dict(gt=ng,tp=tp,fp=fp,fn=ng-tp,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/ng if ng else None,AP50=ap)
        all_tp+=tp;all_fp+=fp;all_gt+=ng
        for im,boxes in gt.items():
            for j,b in enumerate(boxes):
                w,h=b[2]-b[0],b[3]-b[1];short=min(w,h)
                ks=['short_lt16' if short<16 else ('short_16_to32' if short<32 else 'short_ge32')]
                if max(w,h)>1024: ks.append('long_gt1024')
                for k in ks:size_counts[k]['gt']+=1;size_counts[k]['tp']+=j in used[im]
    precision=all_tp/(all_tp+all_fp) if all_tp+all_fp else 0;recall=all_tp/all_gt if all_gt else 0
    ap=[v['AP50'] for v in report.values() if v['AP50'] is not None]
    return dict(images=len(truth),classes=names,iou=threshold,tp=all_tp,fp=all_fp,fn=all_gt-all_tp,
                precision=precision,recall=recall,F1=2*precision*recall/(precision+recall) if precision+recall else 0,
                false_positives_per_image=all_fp/len(truth) if truth else 0,mAP50=float(np.mean(ap)) if ap else None,
                per_class=report,by_gt_geometry={k:dict(v,recall=v['tp']/v['gt'] if v['gt'] else None) for k,v in size_counts.items()},
                metric_scope='Local 101-point AP at configured IoU on exported V4E predictions; not official platform score or full unfiltered detector AP.')
