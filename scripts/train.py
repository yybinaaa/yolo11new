from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    DEFAULT_CONFIG,
    activate_version,
    load_experiment,
    validate_inputs,
    workspace_path,
    write_launch_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用指定源码版本训练 YOLO11")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--version", help="版本目录名；默认读取 active_version")
    parser.add_argument("--resume", type=Path, help="从指定 last.pt 恢复训练")
    parser.add_argument("--dry-run", action="store_true", help="只加载源码、配置和模型，不开始训练")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, config = load_experiment(args.config, args.version)
    data, pretrained, model_yaml = validate_inputs(config)
    activate_version(config["version"], config["version_meta"].get("source_version"))

    from ultralytics import YOLO

    resume = args.resume.expanduser().resolve() if args.resume else None
    if resume and not resume.is_file():
        raise FileNotFoundError(f"恢复权重不存在: {resume}")

    model_config = config["model"]
    if resume:
        model = YOLO(str(resume), task=config["task"])
    elif model_config.get("build_from_yaml"):
        model = YOLO(str(model_yaml), task=config["task"])
        if model_config.get("load_pretrained"):
            model.load(str(pretrained))
    elif model_config.get("load_pretrained"):
        model = YOLO(str(pretrained), task=config["task"])
    else:
        model = YOLO(str(model_yaml), task=config["task"])

    record = write_launch_record(
        "train",
        config,
        {"data": data, "model_yaml": model_yaml, "pretrained": pretrained, "resume": resume},
    )
    print(f"源码版本: {config['version']}")
    print(f"源码位置: {Path(sys.modules['ultralytics'].__file__).resolve().parent}")
    print(f"启动记录: {record}")
    if args.dry_run:
        model.info(verbose=False)
        print("Dry-run 通过：未开始训练。")
        return 0

    params = dict(config["train"])
    runs = workspace_path(config["paths"]["runs"])
    params.update(data=str(data), project=str(runs / config["version"]))
    if resume:
        params["resume"] = str(resume)
    model.train(**params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

