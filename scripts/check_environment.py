from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from common import DEFAULT_CONFIG, activate_version, load_experiment, source_digest, validate_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="检查当前环境和本地 YOLO11 源码")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--version")
    args = parser.parse_args()

    _, config = load_experiment(args.config, args.version)
    data, weights, model_yaml = validate_inputs(config)
    ultralytics = activate_version(config["version"], config["version_meta"].get("source_version"))
    import torch

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"系统: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()} / CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")
    print(f"源码版本: {config['version']} / Ultralytics {ultralytics.__version__}")
    print(f"源码入口: {Path(ultralytics.__file__).resolve()}")
    print(f"源码 SHA256: {source_digest(config['version'])}")
    print(f"模型 YAML: {model_yaml}")
    print(f"预训练权重: {weights}")
    print(f"数据配置: {data}")
    print("环境检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

