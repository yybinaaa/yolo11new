from __future__ import annotations

def starts(length: int, tile: int, overlap: float) -> list[int]:
    if tile <= 0 or length <= 0 or not 0 <= overlap < 1: raise ValueError("非法切片参数")
    if length <= tile: return [0]
    step=max(1, round(tile*(1-overlap)))
    values=list(range(0, length-tile+1, step))
    if values[-1] != length-tile: values.append(length-tile)
    return values

def clip_box(box: tuple[float,float,float,float], window: tuple[int,int,int,int], min_visibility: float=0.3):
    x1,y1,x2,y2=box; wx1,wy1,wx2,wy2=window
    ix1,iy1,ix2,iy2=max(x1,wx1),max(y1,wy1),min(x2,wx2),min(y2,wy2)
    box_w,box_h=max(0,x2-x1),max(0,y2-y1); visible=max(0,ix2-ix1)*max(0,iy2-iy1)
    # A defect larger than one tile can never reach a fraction of its full area.
    # Normalize by the largest portion that can physically fit in this window.
    reference=min(box_w,max(0,wx2-wx1))*min(box_h,max(0,wy2-wy1))
    if reference <= 0 or visible/reference < min_visibility or ix2 <= ix1 or iy2 <= iy1: return None
    return (ix1-wx1,iy1-wy1,ix2-wx1,iy2-wy1)

def restore_box(box: tuple[float,float,float,float], x: int, y: int):
    return (box[0]+x,box[1]+y,box[2]+x,box[3]+y)
