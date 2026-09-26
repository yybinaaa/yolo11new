"""Independent SAHI-style inference entry point for the YOLO11 epoch-93 model.

This file intentionally does not modify or replace the project's original
training and inference scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import _bootstrap  # noqa: F401  (adds the project root to sys.path)
from build_submission import convert_predictions, save_submission
from src.inference.merger import classwise_nms
from src.inference.tiler import restore_box, starts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs/yolo11/yolo11m_compare_100e/weights/epoch93.pt"
DEFAULT_SOURCE = ROOT.parent / "初赛测试集/初赛"
DEFAULT_OUTPUT = ROOT / "outputs/yolo11/sahi_epoch93"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/matplotlib"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 YOLO11 epoch93 执行独立的 SAHI 切片 + 整图推理"
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--full-imgsz", type=int, default=1024)
    parser.add_argument("--no-full-image", action="store_true")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument(
        "--thresholds", type=Path, default=ROOT / "configs/class_thresholds.yaml"
    )
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        help="可选：保存跨切片 NMS 前的候选框，供离线阈值/NMS/融合 A/B 使用",
    )
    parser.add_argument("--submission-name", default="yolo11_epoch93_sahi_submission")
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate(args: argparse.Namespace) -> None:
    args.weights, args.source = resolve(args.weights), resolve(args.source)
    args.output, args.thresholds = resolve(args.output), resolve(args.thresholds)
    if args.candidate_cache is not None:
        args.candidate_cache = resolve(args.candidate_cache)
    for path in (args.weights, args.source, args.thresholds):
        if not path.exists():
            raise FileNotFoundError(path)
    if min(args.tile_size, args.full_imgsz, args.batch, args.max_det) <= 0:
        raise ValueError("tile-size、full-imgsz、batch 和 max-det 必须大于 0")
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap 必须位于 [0, 1)")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("conf 和 iou 必须位于 [0, 1]")


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        files = [source] if source.suffix.lower() in IMAGE_SUFFIXES else []
    else:
        files = sorted(
            p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    if not files:
        raise FileNotFoundError(f"没有找到待推理图片: {source}")
    return files


def append_boxes(
    detections: list[list[float]],
    result: object,
    x_offset: int = 0,
    y_offset: int = 0,
    *,
    candidate_boxes: list[dict[str, object]] | None = None,
    source: str = "tile",
) -> None:
    boxes = result.boxes
    if boxes is None:
        return
    for box, score, class_id in zip(
        boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
    ):
        restored = restore_box(tuple(float(v) for v in box), x_offset, y_offset)
        score_value = float(score)
        class_value = int(class_id)
        detections.append([*restored, score_value, float(class_value)])
        if candidate_boxes is not None:
            candidate = {
                "bbox_xyxy": [float(value) for value in restored],
                "score": score_value,
                "class_id": class_value,
                "source": source,
            }
            if source == "tile":
                candidate["tile_origin"] = [int(x_offset), int(y_offset)]
            candidate_boxes.append(candidate)


def annotate(image: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
    for item in detections:
        x1, y1, x2, y2 = (round(float(v)) for v in item["bbox_xyxy"])
        class_id, score = int(item["class_id"]), float(item["score"])
        color = (
            (37 * class_id + 67) % 206 + 50,
            (97 * class_id + 29) % 206 + 50,
            (17 * class_id + 149) % 206 + 50,
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f'{item["class_name"]} {score:.2f}',
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    args = arguments()
    validate(args)
    args.output.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    threshold_data = yaml.safe_load(args.thresholds.read_text(encoding="utf-8")) or {}
    default_threshold = float(threshold_data.get("default", 0.25))
    class_thresholds = threshold_data.get("classes", {}) or {}
    records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    files = list_images(args.source)
    total_started = time.perf_counter()

    for image_path in tqdm(files, desc="YOLO11 epoch93 SAHI"):
        image_started = time.perf_counter()
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"图片读取失败: {image_path}")
        height, width = image.shape[:2]
        tiles = [
            (image[y : y + args.tile_size, x : x + args.tile_size], x, y)
            for y in starts(height, args.tile_size, args.overlap)
            for x in starts(width, args.tile_size, args.overlap)
        ]

        raw: list[list[float]] = []
        image_candidates: list[dict[str, object]] | None = (
            [] if args.candidate_cache is not None else None
        )
        forward_seconds = 0.0
        for offset in range(0, len(tiles), args.batch):
            chunk = tiles[offset : offset + args.batch]
            started = time.perf_counter()
            results = model.predict(
                [tile for tile, _, _ in chunk],
                imgsz=args.tile_size,
                conf=args.conf,
                batch=args.batch,
                device=args.device,
                max_det=args.max_det,
                verbose=False,
            )
            forward_seconds += time.perf_counter() - started
            for result, (_, x, y) in zip(results, chunk):
                append_boxes(
                    raw,
                    result,
                    x,
                    y,
                    candidate_boxes=image_candidates,
                    source="tile",
                )

        full_image_used = not args.no_full_image and len(tiles) > 1
        if full_image_used:
            started = time.perf_counter()
            result = model.predict(
                image,
                imgsz=args.full_imgsz,
                conf=args.conf,
                device=args.device,
                max_det=args.max_det,
                verbose=False,
            )[0]
            forward_seconds += time.perf_counter() - started
            append_boxes(
                raw,
                result,
                candidate_boxes=image_candidates,
                source="full",
            )

        merged = classwise_nms(
            np.asarray(raw, dtype=float) if raw else np.empty((0, 6), dtype=float), args.iou
        )
        found: list[dict[str, object]] = []
        for x1, y1, x2, y2, score, raw_class_id in merged:
            class_id = int(raw_class_id)
            name = str(model.names[class_id])
            if score < float(class_thresholds.get(name, default_threshold)):
                continue
            found.append(
                {
                    "class_id": class_id,
                    "class_name": name,
                    "score": float(score),
                    "bbox_xyxy": [
                        float(np.clip(x1, 0, width)),
                        float(np.clip(y1, 0, height)),
                        float(np.clip(x2, 0, width)),
                        float(np.clip(y2, 0, height)),
                    ],
                }
            )

        if not args.no_images:
            rendered = annotate(image.copy(), found)
            ok, encoded = cv2.imencode(image_path.suffix, rendered)
            if not ok:
                raise RuntimeError(f"结果图片编码失败: {image_path}")
            encoded.tofile(args.output / image_path.name)
        records.append(
            {
                "image_id": image_path.name,
                "detections": found,
                "timing": {
                    "total": time.perf_counter() - image_started,
                    "forward": forward_seconds,
                    "tiles": len(tiles),
                    "full_image_used": full_image_used,
                },
            }
        )
        if image_candidates is not None:
            candidate_records.append(
                {
                    "image_id": image_path.name,
                    "width": width,
                    "height": height,
                    "candidates": image_candidates,
                    "tiles": len(tiles),
                    "full_image_used": full_image_used,
                }
            )

    raw_path = args.output / "predictions.json"
    raw_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    submission = convert_predictions(records, args.source if args.source.is_dir() else None)
    submission_json = args.output / f"{args.submission_name}.json"
    submission_zip = args.output / f"{args.submission_name}.zip"
    save_submission(submission, submission_json, submission_zip)
    if args.candidate_cache is not None:
        names = (
            {str(key): str(value) for key, value in model.names.items()}
            if isinstance(model.names, dict)
            else {str(index): str(value) for index, value in enumerate(model.names)}
        )
        candidate_payload = {
            "schema_version": 1,
            "weights": str(args.weights),
            "source": str(args.source),
            "class_names": names,
            "settings": {
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "batch": args.batch,
                "candidate_conf": args.conf,
                "full_image": not args.no_full_image,
                "full_imgsz": args.full_imgsz,
                "max_det": args.max_det,
            },
            "images": candidate_records,
        }
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_cache.write_text(
            json.dumps(candidate_payload, ensure_ascii=False), encoding="utf-8"
        )
    summary = {
        "strategy": "SAHI (overlapping tiles + full-image inference + classwise NMS)",
        "weights": str(args.weights),
        "images": len(files),
        "detections": len(submission),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "full_image": not args.no_full_image,
        "total_seconds": time.perf_counter() - total_started,
        "submission_json": str(submission_json),
        "submission_zip": str(submission_zip),
        "candidate_cache": str(args.candidate_cache) if args.candidate_cache else None,
    }
    (args.output / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
