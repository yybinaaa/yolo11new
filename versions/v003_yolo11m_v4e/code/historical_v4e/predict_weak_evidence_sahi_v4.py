"""Weak-Evidence Amplified SAHI V4 for the epoch170 detector.

Candidates below the V3E output threshold are treated as proposals instead of
being submitted directly.  Each selected proposal is observed through a 640px
zoom crop and a 1280px context crop.  The resulting detections are matched back
to their weak seed evidence and safely appended to the V3E anchor submission.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from scripts import _bootstrap  # type: ignore  # noqa: F401
try:
    from build_submission import convert_predictions, save_submission
    from build_epoch170_adaptive_v3_ab import (
        additive_safe_submission,
        sha256,
        validate_zip,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.build_submission import convert_predictions, save_submission
    from scripts.build_epoch170_adaptive_v3_ab import (
        additive_safe_submission,
        sha256,
        validate_zip,
    )
from src.inference.merger import classwise_nms, iou


ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/matplotlib"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="epoch170 弱证据放大 SAHI V4")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--first-pass-cache", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True, help="V3E 官方扁平 JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weak-min", type=float, default=0.05)
    parser.add_argument("--weak-max", type=float, default=0.18)
    parser.add_argument("--zoom-size", type=int, default=640)
    parser.add_argument("--context-size", type=int, default=1280)
    parser.add_argument("--model-imgsz", type=int, default=1024)
    parser.add_argument("--max-seeds-per-image", type=int, default=2)
    parser.add_argument("--cluster-iou", type=float, default=0.25)
    parser.add_argument("--cluster-iom", type=float, default=0.55)
    parser.add_argument("--anchor-overlap-iou", type=float, default=0.30)
    parser.add_argument("--dedupe-center-distance", type=float, default=192.0)
    parser.add_argument("--confirm-iou", type=float, default=0.10)
    parser.add_argument("--confirm-iom", type=float, default=0.50)
    parser.add_argument("--safe-threshold", type=float, default=0.18)
    parser.add_argument("--aggressive-threshold", type=float, default=0.10)
    parser.add_argument("--agreement-floor", type=float, default=0.08)
    parser.add_argument("--agreement-iou", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--novel-iou", type=float, default=0.50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def intersection_over_min(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1, y1 = np.maximum(box_a[:2], box_b[:2])
    x2, y2 = np.minimum(box_a[2:4], box_b[2:4])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    return intersection / max(1e-12, min(area_a, area_b))


def center_of(box: list[float] | np.ndarray) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)


def centered_origin(
    center_x: float, center_y: float, *, width: int, height: int, crop_size: int
) -> tuple[int, int]:
    max_x, max_y = max(0, width - crop_size), max(0, height - crop_size)
    return (
        int(np.clip(round(center_x - crop_size / 2), 0, max_x)),
        int(np.clip(round(center_y - crop_size / 2), 0, max_y)),
    )


def _anchor_index(anchor: list[dict[str, object]]) -> dict[tuple[str, str], list[np.ndarray]]:
    grouped: defaultdict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for item in anchor:
        grouped[(str(item["image_id"]), str(item["category_name"]))].append(
            np.asarray(item["bbox"], dtype=float)
        )
    return dict(grouped)


def _same_evidence_region(
    left: dict[str, object],
    right: dict[str, object],
    *,
    cluster_iou: float,
    cluster_iom: float,
) -> bool:
    if int(left["class_id"]) != int(right["class_id"]):
        return False
    left_box = np.asarray(left["bbox_xyxy"], dtype=float)
    right_box = np.asarray(right["bbox_xyxy"], dtype=float)
    return iou(left_box, right_box) >= cluster_iou or intersection_over_min(
        left_box, right_box
    ) >= cluster_iom


def _cluster_weak_candidates(
    candidates: list[dict[str, object]],
    *,
    cluster_iou: float,
    cluster_iom: float,
) -> list[dict[str, object]]:
    clusters: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda row: float(row["score"]), reverse=True):
        target = None
        for cluster in clusters:
            if _same_evidence_region(
                candidate,
                cluster["representative"],
                cluster_iou=cluster_iou,
                cluster_iom=cluster_iom,
            ):
                target = cluster
                break
        if target is None:
            clusters.append({"representative": candidate, "members": [candidate]})
        else:
            target["members"].append(candidate)

    for cluster in clusters:
        members = cluster["members"]
        views = {
            (
                str(member.get("source", "unknown")),
                tuple(member.get("tile_origin", [])),
            )
            for member in members
        }
        max_score = max(float(member["score"]) for member in members)
        cluster["support"] = len(views)
        cluster["priority"] = max_score + 0.025 * min(3, len(views) - 1)
    return sorted(clusters, key=lambda row: float(row["priority"]), reverse=True)


def build_weak_evidence_plan(
    cache: dict[str, object],
    anchor: list[dict[str, object]],
    *,
    weak_min: float,
    weak_max: float,
    zoom_size: int,
    context_size: int,
    max_seeds_per_image: int,
    cluster_iou: float,
    cluster_iom: float,
    anchor_overlap_iou: float,
    dedupe_center_distance: float,
) -> dict[str, object]:
    names = {int(key): str(value) for key, value in cache["class_names"].items()}
    anchors = _anchor_index(anchor)
    planned_images: list[dict[str, object]] = []
    weak_considered = 0
    weak_after_anchor_filter = 0
    cluster_count = 0

    for image in cache["images"]:
        image_id = str(image["image_id"])
        width, height = int(image["width"]), int(image["height"])
        weak: list[dict[str, object]] = []
        for candidate in image["candidates"]:
            score = float(candidate["score"])
            if not weak_min <= score < weak_max:
                continue
            weak_considered += 1
            class_id = int(candidate["class_id"])
            box = np.asarray(candidate["bbox_xyxy"], dtype=float)
            if any(
                iou(box, anchor_box) >= anchor_overlap_iou
                for anchor_box in anchors.get((image_id, names[class_id]), [])
            ):
                continue
            weak.append(candidate)
        weak_after_anchor_filter += len(weak)
        clusters = _cluster_weak_candidates(
            weak, cluster_iou=cluster_iou, cluster_iom=cluster_iom
        )
        cluster_count += len(clusters)

        selected: list[dict[str, object]] = []
        for cluster in clusters:
            representative = cluster["representative"]
            candidate_center = center_of(representative["bbox_xyxy"])
            nearest = None
            nearest_distance = float("inf")
            for group in selected:
                gx, gy = group["center"]
                distance = math.hypot(candidate_center[0] - gx, candidate_center[1] - gy)
                if distance <= dedupe_center_distance and distance < nearest_distance:
                    nearest, nearest_distance = group, distance

            seed = {
                "bbox_xyxy": [float(value) for value in representative["bbox_xyxy"]],
                "class_id": int(representative["class_id"]),
                "class_name": names[int(representative["class_id"])],
                "score": float(representative["score"]),
                "support": int(cluster["support"]),
                "member_count": len(cluster["members"]),
                "source": str(representative.get("source", "unknown")),
            }
            if nearest is not None:
                nearest["seeds"].append(seed)
                continue
            if len(selected) >= max_seeds_per_image:
                continue
            group_id = f"g{len(selected)}"
            selected.append(
                {
                    "group_id": group_id,
                    "center": [candidate_center[0], candidate_center[1]],
                    "priority": float(cluster["priority"]),
                    "seeds": [seed],
                }
            )

        if not selected:
            continue
        for group in selected:
            cx, cy = (float(value) for value in group["center"])
            views = []
            for view_type, crop_size in (("zoom", zoom_size), ("context", context_size)):
                origin_x, origin_y = centered_origin(
                    cx, cy, width=width, height=height, crop_size=crop_size
                )
                views.append(
                    {
                        "view_id": f"{group['group_id']}_{view_type}",
                        "group_id": group["group_id"],
                        "view_type": view_type,
                        "crop_size": crop_size,
                        "origin": [origin_x, origin_y],
                    }
                )
            group["views"] = views
        planned_images.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "groups": selected,
            }
        )

    return {
        "schema_version": 1,
        "strategy": "Weak-Evidence Amplified dual-scale SAHI V4",
        "settings": {
            "weak_score_range": [weak_min, weak_max],
            "zoom_size": zoom_size,
            "context_size": context_size,
            "max_seeds_per_image": max_seeds_per_image,
            "cluster_iou": cluster_iou,
            "cluster_iom": cluster_iom,
            "anchor_overlap_iou": anchor_overlap_iou,
            "dedupe_center_distance": dedupe_center_distance,
        },
        "source_images": len(cache["images"]),
        "weak_candidates_considered": weak_considered,
        "weak_candidates_after_anchor_filter": weak_after_anchor_filter,
        "weak_clusters": cluster_count,
        "planned_images": len(planned_images),
        "planned_seed_groups": sum(len(image["groups"]) for image in planned_images),
        "planned_views": 2 * sum(len(image["groups"]) for image in planned_images),
        "images": planned_images,
    }


def _append_result(
    output: list[dict[str, object]], result: object, view: dict[str, object]
) -> None:
    boxes = result.boxes
    if boxes is None:
        return
    origin_x, origin_y = (int(value) for value in view["origin"])
    for box, score, class_id in zip(
        boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
    ):
        x1, y1, x2, y2 = (float(value) for value in box)
        output.append(
            {
                "bbox_xyxy": [
                    x1 + origin_x,
                    y1 + origin_y,
                    x2 + origin_x,
                    y2 + origin_y,
                ],
                "score": float(score),
                "class_id": int(class_id),
                "group_id": str(view["group_id"]),
                "view_id": str(view["view_id"]),
                "view_type": str(view["view_type"]),
                "crop_size": int(view["crop_size"]),
                "origin": [origin_x, origin_y],
            }
        )


def run_dual_scale_inference(
    plan: dict[str, object],
    *,
    weights: Path,
    source: Path,
    cache_path: Path,
    model_imgsz: int,
    batch: int,
    conf: float,
    device: str,
    max_det: int,
    resume: bool,
) -> dict[str, object]:
    from ultralytics import YOLO

    completed: dict[str, dict[str, object]] = {}
    elapsed_previous = 0.0
    if resume and cache_path.is_file():
        existing = load_json(cache_path)
        if isinstance(existing, dict) and existing.get("schema_version") == 1:
            completed = {str(row["image_id"]): row for row in existing.get("images", [])}
            elapsed_previous = float(existing.get("elapsed_seconds", 0.0))

    model = YOLO(str(weights))
    started = time.perf_counter()
    processed_since_save = 0
    for item in tqdm(plan["images"], desc="Weak-Evidence SAHI V4"):
        image_id = str(item["image_id"])
        if image_id in completed:
            continue
        image_path = source / image_id
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"图片读取失败: {image_path}")
        views = [view for group in item["groups"] for view in group["views"]]
        candidates: list[dict[str, object]] = []
        for offset in range(0, len(views), batch):
            chunk = views[offset : offset + batch]
            crops = []
            for view in chunk:
                origin_x, origin_y = (int(value) for value in view["origin"])
                crop_size = int(view["crop_size"])
                crops.append(
                    image[
                        origin_y : origin_y + crop_size,
                        origin_x : origin_x + crop_size,
                    ]
                )
            results = model.predict(
                crops,
                imgsz=model_imgsz,
                conf=conf,
                batch=batch,
                device=device,
                max_det=max_det,
                verbose=False,
            )
            for result, view in zip(results, chunk):
                _append_result(candidates, result, view)
        completed[image_id] = {
            "image_id": image_id,
            "width": int(item["width"]),
            "height": int(item["height"]),
            "groups": item["groups"],
            "candidates": candidates,
        }
        processed_since_save += 1
        if processed_since_save % 20 == 0:
            atomic_write_json(
                cache_path,
                {
                    "schema_version": 1,
                    "strategy": plan["strategy"],
                    "weights": str(weights),
                    "settings": plan["settings"],
                    "elapsed_seconds": elapsed_previous + time.perf_counter() - started,
                    "images": list(completed.values()),
                },
            )

    names = (
        {int(key): str(value) for key, value in model.names.items()}
        if isinstance(model.names, dict)
        else {index: str(value) for index, value in enumerate(model.names)}
    )
    payload = {
        "schema_version": 1,
        "strategy": plan["strategy"],
        "weights": str(weights),
        "settings": plan["settings"],
        "class_names": names,
        "elapsed_seconds": elapsed_previous + time.perf_counter() - started,
        "images": list(completed.values()),
    }
    atomic_write_json(cache_path, payload)
    return payload


def detection_matches_seed(
    detection: dict[str, object],
    group: dict[str, object],
    *,
    confirm_iou: float,
    confirm_iom: float,
) -> bool:
    box = np.asarray(detection["bbox_xyxy"], dtype=float)
    class_id = int(detection["class_id"])
    for seed in group["seeds"]:
        if int(seed["class_id"]) != class_id:
            continue
        seed_box = np.asarray(seed["bbox_xyxy"], dtype=float)
        if iou(box, seed_box) >= confirm_iou or intersection_over_min(
            box, seed_box
        ) >= confirm_iom:
            return True
    return False


def adaptive_view_type(group: dict[str, object]) -> str:
    seed = max(group["seeds"], key=lambda row: float(row["score"]))
    x1, y1, x2, y2 = (float(value) for value in seed["bbox_xyxy"])
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    aspect = max(width / height, height / width)
    return "context" if aspect >= 4.0 or max(width, height) >= 320.0 else "zoom"


def confirmed_records(
    cache: dict[str, object],
    *,
    threshold: float,
    nms_iou: float,
    confirm_iou: float,
    confirm_iom: float,
    view_type: str | None = None,
    adaptive: bool = False,
) -> list[dict[str, object]]:
    class_names = {int(key): str(value) for key, value in cache["class_names"].items()}
    records: list[dict[str, object]] = []
    for image in cache["images"]:
        group_map = {str(group["group_id"]): group for group in image["groups"]}
        rows = []
        for detection in image["candidates"]:
            group = group_map[str(detection["group_id"])]
            expected_view = adaptive_view_type(group) if adaptive else view_type
            if expected_view is not None and str(detection["view_type"]) != expected_view:
                continue
            if float(detection["score"]) < threshold:
                continue
            if not detection_matches_seed(
                detection,
                group,
                confirm_iou=confirm_iou,
                confirm_iom=confirm_iom,
            ):
                continue
            rows.append(
                [
                    *[float(value) for value in detection["bbox_xyxy"]],
                    float(detection["score"]),
                    float(detection["class_id"]),
                ]
            )
        raw = np.asarray(rows, dtype=float) if rows else np.empty((0, 6), dtype=float)
        merged = classwise_nms(raw, nms_iou)
        detections = []
        width, height = int(image["width"]), int(image["height"])
        for x1, y1, x2, y2, score, raw_class_id in merged:
            class_id = int(raw_class_id)
            detections.append(
                {
                    "class_name": class_names[class_id],
                    "score": float(score),
                    "bbox_xyxy": [
                        float(np.clip(x1, 0, width)),
                        float(np.clip(y1, 0, height)),
                        float(np.clip(x2, 0, width)),
                        float(np.clip(y2, 0, height)),
                    ],
                }
            )
        records.append({"image_id": image["image_id"], "detections": detections})
    return records


def dual_agreement_records(
    cache: dict[str, object],
    *,
    score_floor: float,
    agreement_iou: float,
    confirm_iou: float,
    confirm_iom: float,
    nms_iou: float,
) -> list[dict[str, object]]:
    class_names = {int(key): str(value) for key, value in cache["class_names"].items()}
    records: list[dict[str, object]] = []
    for image in cache["images"]:
        group_map = {str(group["group_id"]): group for group in image["groups"]}
        by_group: defaultdict[str, defaultdict[str, list[dict[str, object]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for detection in image["candidates"]:
            group_id = str(detection["group_id"])
            if float(detection["score"]) < score_floor:
                continue
            if detection_matches_seed(
                detection,
                group_map[group_id],
                confirm_iou=confirm_iou,
                confirm_iom=confirm_iom,
            ):
                by_group[group_id][str(detection["view_type"])].append(detection)

        fused_rows = []
        for views in by_group.values():
            zoom = sorted(views.get("zoom", []), key=lambda row: float(row["score"]), reverse=True)
            context = sorted(
                views.get("context", []), key=lambda row: float(row["score"]), reverse=True
            )
            used_context: set[int] = set()
            for zoom_item in zoom:
                zoom_box = np.asarray(zoom_item["bbox_xyxy"], dtype=float)
                best_index = None
                best_overlap = agreement_iou
                for index, context_item in enumerate(context):
                    if index in used_context or int(context_item["class_id"]) != int(
                        zoom_item["class_id"]
                    ):
                        continue
                    context_box = np.asarray(context_item["bbox_xyxy"], dtype=float)
                    overlap = max(
                        iou(zoom_box, context_box),
                        intersection_over_min(zoom_box, context_box),
                    )
                    if overlap >= best_overlap:
                        best_index, best_overlap = index, overlap
                if best_index is None:
                    continue
                used_context.add(best_index)
                context_item = context[best_index]
                context_box = np.asarray(context_item["bbox_xyxy"], dtype=float)
                scores = np.asarray(
                    [float(zoom_item["score"]), float(context_item["score"])], dtype=float
                )
                fused_box = np.average(
                    np.stack((zoom_box, context_box)), axis=0, weights=scores
                )
                fused_rows.append(
                    [*fused_box.tolist(), float(scores.mean()), float(zoom_item["class_id"])]
                )

        raw = (
            np.asarray(fused_rows, dtype=float)
            if fused_rows
            else np.empty((0, 6), dtype=float)
        )
        merged = classwise_nms(raw, nms_iou)
        width, height = int(image["width"]), int(image["height"])
        detections = []
        for x1, y1, x2, y2, score, raw_class_id in merged:
            class_id = int(raw_class_id)
            detections.append(
                {
                    "class_name": class_names[class_id],
                    "score": float(score),
                    "bbox_xyxy": [
                        float(np.clip(x1, 0, width)),
                        float(np.clip(y1, 0, height)),
                        float(np.clip(x2, 0, width)),
                        float(np.clip(y2, 0, height)),
                    ],
                }
            )
        records.append({"image_id": image["image_id"], "detections": detections})
    return records


def write_variant(
    output: Path,
    *,
    stem: str,
    description: str,
    anchor: list[dict[str, object]],
    branch: list[dict[str, object]],
    novel_iou: float,
) -> dict[str, object]:
    submission, additions = additive_safe_submission(anchor, branch, novel_iou=novel_iou)
    if submission[: len(anchor)] != anchor:
        raise RuntimeError(f"{stem} 未完整保留 V3E anchor，拒绝生成")
    json_path = output / f"{stem}.json"
    zip_path = output / f"{stem}.zip"
    save_submission(submission, json_path, zip_path)
    validate_zip(zip_path, json_path)
    return {
        "stem": stem,
        "description": description,
        "detections": len(submission),
        "anchor_detections": len(anchor),
        "additive_novel_detections": additions,
        "images_with_detections": len({str(item["image_id"]) for item in submission}),
        "anchor_prefix_preserved": True,
        "json": str(json_path),
        "zip": str(zip_path),
        "json_sha256": sha256(json_path),
        "zip_sha256": sha256(zip_path),
        "zip_validation": "passed",
    }


def main() -> None:
    args = arguments()
    probability_names = (
        "weak_min",
        "weak_max",
        "cluster_iou",
        "cluster_iom",
        "anchor_overlap_iou",
        "confirm_iou",
        "confirm_iom",
        "safe_threshold",
        "aggressive_threshold",
        "agreement_floor",
        "agreement_iou",
        "nms_iou",
        "novel_iou",
        "conf",
    )
    for name in probability_names:
        if not 0 <= float(getattr(args, name)) <= 1:
            raise ValueError(f"{name} 必须位于 [0, 1]")
    if args.weak_min >= args.weak_max:
        raise ValueError("weak_min 必须小于 weak_max")
    if min(
        args.zoom_size,
        args.context_size,
        args.model_imgsz,
        args.max_seeds_per_image,
        args.batch,
        args.max_det,
    ) <= 0:
        raise ValueError("尺寸、seed 数量、batch 和 max_det 必须大于 0")

    weights = resolve(args.weights)
    source = resolve(args.source)
    first_path = resolve(args.first_pass_cache)
    anchor_path = resolve(args.anchor)
    output = resolve(args.output)
    for path in (weights, source, first_path, anchor_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)

    first_pass = load_json(first_path)
    anchor = load_json(anchor_path)
    if not isinstance(first_pass, dict) or first_pass.get("schema_version") != 1:
        raise ValueError("一阶段候选缓存格式不受支持")
    if not isinstance(anchor, list):
        raise TypeError("V3E anchor 必须是官方扁平 JSON 列表")

    plan = build_weak_evidence_plan(
        first_pass,
        anchor,
        weak_min=args.weak_min,
        weak_max=args.weak_max,
        zoom_size=args.zoom_size,
        context_size=args.context_size,
        max_seeds_per_image=args.max_seeds_per_image,
        cluster_iou=args.cluster_iou,
        cluster_iom=args.cluster_iom,
        anchor_overlap_iou=args.anchor_overlap_iou,
        dedupe_center_distance=args.dedupe_center_distance,
    )
    plan_path = output / "weak_evidence_sahi_v4_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    plan_summary = {key: value for key, value in plan.items() if key != "images"}
    if args.plan_only:
        print(json.dumps(plan_summary, ensure_ascii=False, indent=2))
        return

    cache_path = output / "weak_evidence_sahi_v4_candidates.json"
    inference = run_dual_scale_inference(
        plan,
        weights=weights,
        source=source,
        cache_path=cache_path,
        model_imgsz=args.model_imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device,
        max_det=args.max_det,
        resume=args.resume,
    )

    variant_specs = [
        (
            "01_V4A_zoom640_t018_safe_add_submission",
            "640 zoom 视野确认，threshold=0.18，安全追加到 V3E",
            confirmed_records(
                inference,
                threshold=args.safe_threshold,
                nms_iou=args.nms_iou,
                confirm_iou=args.confirm_iou,
                confirm_iom=args.confirm_iom,
                view_type="zoom",
            ),
        ),
        (
            "02_V4B_context1280_t018_safe_add_submission",
            "1280 context 视野确认，threshold=0.18，安全追加到 V3E",
            confirmed_records(
                inference,
                threshold=args.safe_threshold,
                nms_iou=args.nms_iou,
                confirm_iou=args.confirm_iou,
                confirm_iom=args.confirm_iom,
                view_type="context",
            ),
        ),
        (
            "03_V4C_shape_adaptive_t018_safe_add_submission",
            "按弱候选形态选择 zoom/context，threshold=0.18，安全追加",
            confirmed_records(
                inference,
                threshold=args.safe_threshold,
                nms_iou=args.nms_iou,
                confirm_iou=args.confirm_iou,
                confirm_iom=args.confirm_iom,
                adaptive=True,
            ),
        ),
        (
            "04_V4D_dual_view_agreement_t008_safe_add_submission",
            "zoom/context 双视野同类一致，floor=0.08，加权框融合后安全追加",
            dual_agreement_records(
                inference,
                score_floor=args.agreement_floor,
                agreement_iou=args.agreement_iou,
                confirm_iou=args.confirm_iou,
                confirm_iom=args.confirm_iom,
                nms_iou=args.nms_iou,
            ),
        ),
        (
            "05_V4E_dual_view_any_t010_safe_add_submission",
            "任一双尺度视野确认，threshold=0.10，Recall 激进安全追加",
            confirmed_records(
                inference,
                threshold=args.aggressive_threshold,
                nms_iou=args.nms_iou,
                confirm_iou=args.confirm_iou,
                confirm_iom=args.confirm_iom,
            ),
        ),
    ]

    outputs = []
    for stem, description, grouped in variant_specs:
        outputs.append(
            write_variant(
                output,
                stem=stem,
                description=description,
                anchor=anchor,
                branch=convert_predictions(grouped),
                novel_iou=args.novel_iou,
            )
        )

    manifest = {
        "strategy": plan["strategy"],
        "weights": str(weights),
        "source": str(source),
        "first_pass_cache": str(first_path),
        "anchor": str(anchor_path),
        "anchor_platform_metrics": {
            "recall": 0.8229,
            "precision": 0.3183,
            "f1": 0.4590,
            "mAP@0.5": 0.4945,
            "TP": 818,
            "FP": 1752,
            "FN": 176,
            "score": 82.29,
        },
        "plan": plan_summary,
        "model_imgsz": args.model_imgsz,
        "candidate_conf": args.conf,
        "inference_seconds": inference.get("elapsed_seconds"),
        "inference_candidates": sum(
            len(image["candidates"]) for image in inference["images"]
        ),
        "confirmation": {
            "confirm_iou": args.confirm_iou,
            "confirm_iom": args.confirm_iom,
            "safe_threshold": args.safe_threshold,
            "aggressive_threshold": args.aggressive_threshold,
            "agreement_floor": args.agreement_floor,
            "agreement_iou": args.agreement_iou,
            "nms_iou": args.nms_iou,
            "novel_iou": args.novel_iou,
        },
        "outputs": outputs,
    }
    manifest_path = output / "weak_evidence_sahi_v4_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
