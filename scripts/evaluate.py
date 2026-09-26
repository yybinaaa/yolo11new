from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from common import (
    DEFAULT_CONFIG,
    activate_version,
    load_experiment,
    newest_trained_weights,
    validate_inputs,
    workspace_path,
    write_launch_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用指定源码版本评测 YOLO11")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--version", help="版本目录名；默认读取 active_version")
    parser.add_argument("--weights", type=Path, help="待评测权重；默认自动寻找该版本最新训练权重")
    parser.add_argument("--dry-run", action="store_true", help="只加载源码、配置和权重，不执行评测")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, config = load_experiment(args.config, args.version)
    data, _, model_yaml = validate_inputs(config, require_pretrained=False)
    weights = args.weights.expanduser().resolve() if args.weights else newest_trained_weights(config)
    if not weights or not weights.is_file():
        raise FileNotFoundError(
            "没有找到评测权重。请先训练，或使用 --weights 指定 best.pt/last.pt。"
        )

    activate_version(config["version"], config["version_meta"].get("source_version"))
    from ultralytics import YOLO

    model = YOLO(str(weights), task=config["task"])
    dataset = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    if dataset.get("validation_is_train"):
        print("警告：当前 data.yaml 的 val 与 train 相同，结果只能用于流程检查，不能作为独立泛化指标。")

    record = write_launch_record(
        "evaluate",
        config,
        {"data": data, "model_yaml": model_yaml, "weights": weights},
    )
    print(f"源码版本: {config['version']}")
    print(f"源码位置: {Path(sys.modules['ultralytics'].__file__).resolve().parent}")
    print(f"启动记录: {record}")
    if args.dry_run:
        model.info(verbose=False)
        print("Dry-run 通过：未执行评测。")
        return 0

    params = dict(config["evaluate"])
    params.pop("weights", None)
    runs = workspace_path(config["paths"]["runs"])
    params.update(data=str(data), project=str(runs / config["version"]))
    metrics = model.val(**params)
    print(f"评测输出: {metrics.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
