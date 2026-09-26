from __future__ import annotations
import numpy as np

def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1,y1=np.maximum(a[:2],b[:2]); x2,y2=np.minimum(a[2:4],b[2:4]); inter=max(0,x2-x1)*max(0,y2-y1)
    return float(inter / max(1e-12, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter))

def classwise_nms(detections: np.ndarray, threshold: float=0.5) -> np.ndarray:
    if len(detections)==0: return np.empty((0,6),dtype=float)
    kept=[]
    for cls in np.unique(detections[:,5]):
        part=detections[detections[:,5]==cls]; order=np.argsort(-part[:,4])
        while len(order):
            idx=order[0]; kept.append(part[idx]); order=np.array([j for j in order[1:] if iou(part[idx],part[j])<=threshold],dtype=int)
    return np.asarray(kept)

