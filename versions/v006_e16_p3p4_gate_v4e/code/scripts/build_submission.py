from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_names(image_dir: Path | None) -> dict[str, str]:
    """Return a stem -> full filename mapping for the official test images."""
    if image_dir is None:
        return {}
    if not image_dir.is_dir():
        raise FileNotFoundError(f"测试图片目录不存在: {image_dir}")

    names: dict[str, str] = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in names and names[path.stem] != path.name:
                raise ValueError(f"测试集中存在同名但扩展名不同的图片: {path.stem}")
            names[path.stem] = path.name
    return names


def _official_image_id(value: object, names: dict[str, str]) -> str:
    image_id = Path(str(value)).name
    if Path(image_id).suffix:
        return image_id
    if image_id in names:
        return names[image_id]
    # The preliminary test set consists of JPG files. This fallback also keeps
    # conversion usable when --image-dir is omitted.
    return f"{image_id}.jpg"


def _official_item(image_id: str, detection: dict) -> dict:
    category_name = detection.get("category_name", detection.get("class_name"))
    bbox = detection.get("bbox", detection.get("bbox_xyxy"))
    score = detection.get("score")

    if not isinstance(category_name, str) or not category_name:
        raise ValueError(f"检测记录缺少 category_name/class_name: {detection}")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"检测记录 bbox 必须包含 4 个坐标: {detection}")

    coords = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in coords):
        raise ValueError(f"bbox 包含非有限数值: {bbox}")
    x1, y1, x2, y2 = [int(round(value)) for value in coords]
    if min(x1, y1) < 0 or x2 < x1 or y2 < y1:
        raise ValueError(f"bbox 坐标无效: {[x1, y1, x2, y2]}")

    confidence = float(score)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"score 必须位于 [0, 1]: {score}")

    return {
        "image_id": image_id,
        "category_name": category_name,
        "bbox": [x1, y1, x2, y2],
        "score": confidence,
    }


def convert_predictions(data: list, image_dir: Path | None = None) -> list[dict]:
    """Convert grouped inference output (or validate flat output) to official format."""
    if not isinstance(data, list):
        raise TypeError("提交内容必须是 JSON 列表")

    names = _image_names(image_dir)
    submission: list[dict] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict) or "image_id" not in row:
            raise ValueError(f"Submission item #{index} missing image_id")
        image_id = _official_image_id(row["image_id"], names)

        if "detections" in row:
            detections = row["detections"]
            if not isinstance(detections, list):
                raise ValueError(f"图片 {image_id} 的 detections 必须是列表")
            submission.extend(_official_item(image_id, detection) for detection in detections)
        else:
            submission.append(_official_item(image_id, row))
    return submission


def save_submission(submission: list[dict], output: Path, zip_output: Path | None = None) -> None:
    """Write an official JSON submission, validation report, and optional ZIP."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")

    report = output.with_name(f"{output.stem}_validation_report.md")
    report.write_text(
        "# 提交文件校验\n\n"
        f"- 检测框数量：{len(submission)}\n"
        f"- 涉及图片数量：{len({item['image_id'] for item in submission})}\n"
        "- 格式：官方扁平 JSON（image_id/category_name/bbox/score）\n"
        "- 校验：通过\n",
        encoding="utf-8",
    )

    if zip_output:
        zip_output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_output, "w", ZIP_DEFLATED) as archive:
            archive.write(output, arcname=output.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成比赛要求的扁平目标检测 JSON 提交文件")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.json"))
    parser.add_argument("--image-dir", type=Path, help="测试图片目录，用于恢复 image_id 的真实扩展名")
    parser.add_argument("--zip-output", type=Path, help="可选：同时生成只包含提交 JSON 的 ZIP")
    args = parser.parse_args()

    source = json.loads(args.predictions.read_text(encoding="utf-8"))
    submission = convert_predictions(source, args.image_dir)

    save_submission(submission, args.output, args.zip_output)

    print(f"提交 JSON: {args.output.resolve()}")
    print(f"检测框数量: {len(submission)}")
    if args.zip_output:
        print(f"提交 ZIP: {args.zip_output.resolve()}")


if __name__ == "__main__":
    main()
