from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE_ROOT / "configs" / "experiment.yaml"


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge update into a copied base dictionary."""
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_experiment(config_path: Path, version: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    selected = version or raw.get("active_version")
    versions = raw.get("versions", {})
    if not selected or selected not in versions:
        available = ", ".join(sorted(versions)) or "无"
        raise KeyError(f"未知版本 {selected!r}；可用版本: {available}")

    resolved = deep_merge(raw.get("defaults", {}), versions[selected].get("overrides", {}))
    resolved["version"] = selected
    resolved["version_meta"] = copy.deepcopy(versions[selected])
    resolved["paths"] = copy.deepcopy(raw.get("paths", {}))
    resolved["config_path"] = str(config_path)
    return raw, resolved


def workspace_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def version_root(version: str) -> Path:
    root = (WORKSPACE_ROOT / "versions" / version).resolve()
    if not (root / "ultralytics" / "__init__.py").is_file():
        raise FileNotFoundError(f"版本源码不完整: {root}")
    return root


def activate_version(version: str, expected_source_version: str | None = None):
    """Put the selected vendored source first on sys.path, then import it."""
    root = version_root(version)
    existing = sys.modules.get("ultralytics")
    if existing is not None:
        loaded = Path(existing.__file__).resolve()
        if root not in loaded.parents:
            raise RuntimeError(f"ultralytics 已从其他位置加载，无法切换版本: {loaded}")

    sys.path.insert(0, str(root))
    paths = (WORKSPACE_ROOT / "artifacts").resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(paths / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(paths / "matplotlib"))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    import ultralytics

    loaded = Path(ultralytics.__file__).resolve()
    if root not in loaded.parents:
        raise RuntimeError(f"没有加载所选版本的本地源码，实际位置: {loaded}")
    if expected_source_version and ultralytics.__version__ != str(expected_source_version):
        raise RuntimeError(
            f"源码版本不一致: config={expected_source_version}, source={ultralytics.__version__}"
        )
    return ultralytics


def validate_inputs(
    config: dict[str, Any], *, require_pretrained: bool = True
) -> tuple[Path, Path | None, Path]:
    paths = config["paths"]
    data = workspace_path(paths["data"])
    pretrained = workspace_path(paths.get("pretrained_weights"))
    root = version_root(config["version"])
    model_definition = str(config["model"]["definition"]).format(
        scale=config["model"].get("scale", "m")
    )
    model_yaml = (root / model_definition).resolve()
    if not data or not data.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data}")
    if not model_yaml.is_file():
        raise FileNotFoundError(f"模型结构配置不存在: {model_yaml}")
    if (
        require_pretrained
        and config["model"].get("load_pretrained")
        and (not pretrained or not pretrained.is_file())
    ):
        raise FileNotFoundError(f"预训练权重不存在: {pretrained}")
    return data, pretrained, model_yaml


def source_digest(version: str) -> str:
    digest = hashlib.sha256()
    root = version_root(version)
    for path in sorted((root / "ultralytics").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_launch_record(kind: str, config: dict[str, Any], extra: dict[str, Any]) -> Path:
    artifact_root = workspace_path(config["paths"]["artifacts"])
    assert artifact_root is not None
    target = artifact_root / "launches" / config["version"]
    target.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record = {
        "kind": kind,
        "created_at": datetime.now().isoformat(),
        "source_sha256": source_digest(config["version"]),
        "resolved_config": config,
        **extra,
    }
    output = target / f"{timestamp}_{kind}.json"
    output.write_text(json.dumps(json_ready(record), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def newest_trained_weights(config: dict[str, Any]) -> Path | None:
    configured = workspace_path(config["evaluate"].get("weights"))
    if configured:
        return configured
    runs = workspace_path(config["paths"]["runs"])
    assert runs is not None
    candidates: list[Path] = []
    for filename in ("best.pt", "last.pt"):
        candidates.extend((runs / config["version"]).glob(f"train*/weights/{filename}"))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    return None
