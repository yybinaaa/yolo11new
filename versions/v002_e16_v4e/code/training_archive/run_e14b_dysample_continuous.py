"""Prepare and run the isolated E14B DySample continuation experiment.

The default action is preparation/audit only.  A long training run starts only
when ``--train`` is explicitly supplied.  The source checkpoint and the
existing Control run are always treated as read-only inputs.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "E14B_dysample_continuous_epoch170_to200.yaml"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep caches created by this entry point inside the project.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "artifacts" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / "matplotlib"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


class ContractError(RuntimeError):
    """Raised when a checkpoint no longer matches the experiment contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "E14B DySample 连续训练。默认只准备和审计分支 checkpoint；"
            "只有 --train 会启动完整训练。"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="E14B 实验配置文件",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="仅准备和审计分支 checkpoint（默认）",
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="使用 0.2%% 数据只跑 epoch171，验证恢复、前反向和保存",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="显式启动正式实验，从 epoch170 连续训练到 epoch200",
    )
    return parser.parse_args(argv)


def selected_mode(args: argparse.Namespace) -> str:
    if args.train:
        return "train"
    if args.smoke_test:
        return "smoke"
    return "prepare"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_experiment_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"实验配置不存在: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("experiment_id") != "E14B":
        raise ContractError("配置 experiment_id 必须是 E14B")

    paths = config.get("paths") or {}
    for key in (
        "source_checkpoint",
        "control_checkpoint",
        "model_yaml",
        "dysample_impl",
        "data_yaml",
    ):
        candidate = resolve_project_path(paths.get(key, ""))
        if not candidate.is_file():
            raise FileNotFoundError(f"配置路径不存在 ({key}): {candidate}")

    source = config.get("source_contract") or {}
    if int(source.get("completed_epoch", -1)) != 170:
        raise ContractError("E14B 必须从已完成的 epoch170 开始")
    if int(source.get("checkpoint_epoch_index", -1)) != 169:
        raise ContractError("checkpoint 内部 epoch 必须是零基的 169")

    scheduler = config.get("scheduler") or {}
    if (
        int(scheduler.get("anchor_epoch", -1)) != 150
        or int(scheduler.get("schedule_horizon_epoch", -1)) != 250
    ):
        raise ContractError("学习率阶段必须保持 Control 的 150→250 锚点")
    if int((config.get("training") or {}).get("stop_epoch", -1)) != 200:
        raise ContractError("正式实验停止点必须是 epoch200")
    if not math.isclose(
        float(scheduler.get("final_lr_factor", -1.0)), 0.20, rel_tol=0, abs_tol=1e-12
    ):
        raise ContractError("学习率阶段的 final_lr_factor 必须保持 0.20")

    architecture = config.get("architecture_contract") or {}
    if (
        list(architecture.get("replacement_layers") or []) != [11, 14]
        or architecture.get("module") != "DySample"
        or architecture.get("style") != "lp"
        or int(architecture.get("scale", -1)) != 2
        or int(architecture.get("groups", -1)) != 4
    ):
        raise ContractError("DySample 架构合同必须固定为 layers 11/14、lp、scale=2、groups=4")

    data_config = yaml.safe_load(
        resolve_project_path(paths["data_yaml"]).read_text(encoding="utf-8")
    ) or {}
    if not data_config.get("validation_is_train"):
        raise ContractError("当前实验约定 data.yaml 明确标记 validation_is_train: true")
    if bool((config.get("training") or {}).get("validation_enabled", True)):
        raise ContractError("没有独立验证集时不得开启验证")
    return config


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def continuation_lr_factor(
    scheduler_epoch: int,
    start_epoch: int = 150,
    schedule_horizon_epoch: int = 250,
    final_factor: float = 0.20,
) -> float:
    """Return the exact E13 continuation multiplier for a scheduler epoch."""
    total_steps = schedule_horizon_epoch - start_epoch
    if total_steps <= 0:
        raise ValueError("schedule_horizon_epoch 必须大于 start_epoch")
    if not 0.0 < final_factor <= 1.0:
        raise ValueError("final_factor 必须位于 (0, 1]")
    completed_steps = min(max(scheduler_epoch - start_epoch + 1, 0), total_steps)
    progress = completed_steps / total_steps
    return final_factor + 0.5 * (1.0 - final_factor) * (
        1.0 + math.cos(math.pi * progress)
    )


def canonical_musgd_group_names(model: Any) -> list[list[str]]:
    """Reproduce Ultralytics 8.4.98 MuSGD grouping, retaining exact order."""
    import torch.nn as nn
    from ultralytics.utils.torch_utils import unwrap_model

    groups: list[dict[str, Any]] = [{}, {}, {}, {}]
    norm_types = tuple(value for key, value in nn.__dict__.items() if "Norm" in key)
    for module_name, module in unwrap_model(model).named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if parameter.ndim >= 2:
                groups[3][fullname] = parameter
            elif "bias" in fullname:
                groups[2][fullname] = parameter
            elif isinstance(module, norm_types) or "logit_scale" in fullname:
                groups[1][fullname] = parameter
            else:
                groups[0][fullname] = parameter

    high_lr = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|SemanticSegment")
    result: list[list[str]] = []
    for group in groups:
        result.append([name for name in group if high_lr.search(name)])
        result.append([name for name in group if not high_lr.search(name)])
    return result


def parameter_shapes(model: Any) -> dict[str, list[int]]:
    from ultralytics.utils.torch_utils import unwrap_model

    return {
        name: list(parameter.shape)
        for name, parameter in unwrap_model(model).named_parameters()
    }


def _flatten(groups: Iterable[Iterable[str]]) -> list[str]:
    return [name for group in groups for name in group]


def _assert_unique_group_names(groups: Sequence[Sequence[str]], label: str) -> None:
    flat = _flatten(groups)
    duplicates = sorted({name for name in flat if flat.count(name) > 1})
    if duplicates:
        raise ContractError(f"{label} 参数分组包含重复名称: {duplicates[:5]}")


def remap_optimizer_state_dict(
    source_optimizer_state: dict[str, Any],
    source_group_names: Sequence[Sequence[str]],
    source_shapes: dict[str, Sequence[int]],
    target_optimizer: Any,
    target_group_names: Sequence[Sequence[str]],
    target_model: Any,
    expected_new_parameter_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map source optimizer state by full parameter name, never by position alone."""
    from ultralytics.utils.torch_utils import unwrap_model

    source_groups = source_optimizer_state.get("param_groups") or []
    target_template = target_optimizer.state_dict()
    target_groups = target_template.get("param_groups") or []
    if len(source_groups) != 8 or len(target_groups) != 8:
        raise ContractError(
            f"MuSGD 必须有 8 个参数组，得到 source={len(source_groups)}, target={len(target_groups)}"
        )
    if len(source_group_names) != 8 or len(target_group_names) != 8:
        raise ContractError("参数名称分组必须与 8 个 MuSGD 参数组一一对应")
    _assert_unique_group_names(source_group_names, "source")
    _assert_unique_group_names(target_group_names, "target")

    source_name_to_id: dict[str, int] = {}
    source_name_to_group: dict[str, int] = {}
    target_name_to_id: dict[str, int] = {}
    target_name_to_group: dict[str, int] = {}
    for index, (saved_group, names) in enumerate(zip(source_groups, source_group_names)):
        ids = list(saved_group.get("params") or [])
        if len(ids) != len(names):
            raise ContractError(
                f"source 第 {index} 组 ID/名称数量不符: {len(ids)} != {len(names)}"
            )
        for parameter_id, name in zip(ids, names):
            source_name_to_id[name] = int(parameter_id)
            source_name_to_group[name] = index
    for index, (saved_group, names) in enumerate(zip(target_groups, target_group_names)):
        ids = list(saved_group.get("params") or [])
        if len(ids) != len(names):
            raise ContractError(
                f"target 第 {index} 组 ID/名称数量不符: {len(ids)} != {len(names)}"
            )
        for parameter_id, name in zip(ids, names):
            target_name_to_id[name] = int(parameter_id)
            target_name_to_group[name] = index

    source_names = set(source_name_to_id)
    target_names = set(target_name_to_id)
    missing_shared = sorted(source_names - target_names)
    actual_new = sorted(target_names - source_names)
    expected_new = sorted(expected_new_parameter_names)
    if missing_shared:
        raise ContractError(f"目标模型缺少 source 参数: {missing_shared[:8]}")
    if actual_new != expected_new:
        raise ContractError(
            f"新增参数不符合单变量实验，expected={expected_new}, actual={actual_new}"
        )

    target_parameters = dict(unwrap_model(target_model).named_parameters())
    for name in sorted(source_names):
        if source_name_to_group[name] != target_name_to_group[name]:
            raise ContractError(
                f"共享参数跨组错位: {name}, source={source_name_to_group[name]}, "
                f"target={target_name_to_group[name]}"
            )
        expected_shape = list(source_shapes.get(name, []))
        actual_shape = list(target_parameters[name].shape)
        if expected_shape != actual_shape:
            raise ContractError(
                f"共享参数形状变化: {name}, source={expected_shape}, target={actual_shape}"
            )

    remapped_groups: list[dict[str, Any]] = []
    for source_group, target_group in zip(source_groups, target_groups):
        group = deepcopy(source_group)
        group["params"] = list(target_group["params"])
        remapped_groups.append(group)

    source_state = source_optimizer_state.get("state") or {}
    remapped_state: dict[int, Any] = {}
    mapped_state_names: list[str] = []
    for name in _flatten(source_group_names):
        source_id = source_name_to_id[name]
        if source_id not in source_state:
            continue
        target_id = target_name_to_id[name]
        remapped_state[target_id] = deepcopy(source_state[source_id])
        mapped_state_names.append(name)

    report = {
        "source_parameter_count": len(source_names),
        "target_parameter_count": len(target_names),
        "mapped_parameter_count": len(source_names),
        "mapped_state_count": len(remapped_state),
        "source_state_count": len(source_state),
        "new_parameter_names": actual_new,
        "source_group_sizes": [len(group) for group in source_group_names],
        "target_group_sizes": [len(group) for group in target_group_names],
        "source_parameters_without_state": sorted(source_names - set(mapped_state_names)),
    }
    if len(remapped_state) != len(source_state):
        raise ContractError(
            f"优化器状态未完整映射: {len(remapped_state)} != {len(source_state)}"
        )
    return {"state": remapped_state, "param_groups": remapped_groups}, report


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    if path.exists():
        path.unlink()
    temporary.replace(path)


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    from ultralytics.utils.patches import torch_save

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有分支 checkpoint: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch_save(payload, temporary)
        if destination.exists():
            raise FileExistsError(f"写入期间目标已出现，拒绝覆盖: {destination}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def experiment_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    source_hash = str(config["source_contract"]["sha256"]).upper()
    artifact_dir = resolve_project_path(paths["artifact_dir"])
    prepared = (
        artifact_dir
        / "prepared"
        / f"epoch170_dysample_branch_{source_hash[:8]}.pt"
    )
    return {
        "source": resolve_project_path(paths["source_checkpoint"]),
        "control": resolve_project_path(paths["control_checkpoint"]),
        "model_yaml": resolve_project_path(paths["model_yaml"]),
        "dysample_impl": resolve_project_path(paths["dysample_impl"]),
        "data_yaml": resolve_project_path(paths["data_yaml"]),
        "artifact_dir": artifact_dir,
        "prepared": prepared,
        "run_project": resolve_project_path(paths["run_project"]),
    }


def assert_static_contract(config: dict[str, Any]) -> dict[str, Any]:
    import ultralytics

    paths = experiment_paths(config)
    expected_version = str(config["runtime_contract"]["ultralytics_version"])
    if ultralytics.__version__ != expected_version:
        raise ContractError(
            f"Ultralytics 版本漂移: expected={expected_version}, actual={ultralytics.__version__}"
        )
    model_hash = sha256_file(paths["model_yaml"])
    impl_hash = sha256_file(paths["dysample_impl"])
    if model_hash != str(config["architecture_contract"]["model_yaml_sha256"]).upper():
        raise ContractError("DySample model YAML 的 SHA256 与实验合同不一致")
    if impl_hash != str(config["architecture_contract"]["dysample_impl_sha256"]).upper():
        raise ContractError("DySample 实现文件的 SHA256 与实验合同不一致")

    full_run = paths["run_project"] / str(config["training"]["run_name"])
    smoke_run = paths["run_project"] / str(config["smoke_test"]["run_name"])
    write_roots = (paths["artifact_dir"], full_run, smoke_run)
    if len({root.resolve() for root in write_roots}) != len(write_roots):
        raise ContractError("artifact、full run 与 smoke run 目录必须互相隔离")
    for input_path in (
        paths["source"],
        paths["control"],
        paths["model_yaml"],
        paths["dysample_impl"],
        paths["data_yaml"],
    ):
        for write_root in write_roots:
            if input_path == write_root or input_path.is_relative_to(write_root):
                raise ContractError(
                    f"只读输入位于实验写目录内: input={input_path}, output={write_root}"
                )
    return {
        "ultralytics_version": ultralytics.__version__,
        "model_yaml_sha256": model_hash,
        "dysample_impl_sha256": impl_hash,
        "data_yaml_sha256": sha256_file(paths["data_yaml"]),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def assert_source_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    paths = experiment_paths(config)
    source = paths["source"]
    contract = config["source_contract"]
    actual_size = source.stat().st_size
    actual_hash = sha256_file(source)
    expected_hash = str(contract["sha256"]).upper()
    if actual_hash != expected_hash:
        raise ContractError(
            f"source checkpoint SHA256 不符，expected={expected_hash}, actual={actual_hash}"
        )
    if actual_size != int(contract["size_bytes"]):
        raise ContractError(
            f"source checkpoint 大小不符，expected={contract['size_bytes']}, actual={actual_size}"
        )
    return {
        "path": str(source),
        "sha256": actual_hash,
        "size_bytes": actual_size,
        "mtime_ns": source.stat().st_mtime_ns,
    }


def _raw_checkpoint(path: Path) -> dict[str, Any]:
    from src.models.ultralytics_registry import register_custom_modules
    from ultralytics.utils.patches import torch_load

    register_custom_modules()
    checkpoint = torch_load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ContractError(f"checkpoint 不是字典格式: {path}")
    return checkpoint


def assert_control_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    path = experiment_paths(config)["control"]
    contract = config["control_contract"]
    actual_hash = sha256_file(path)
    actual_size = path.stat().st_size
    if actual_hash != str(contract["sha256"]).upper():
        raise ContractError("Control epoch200 checkpoint 的 SHA256 与合同不一致")
    if actual_size != int(contract["size_bytes"]):
        raise ContractError("Control epoch200 checkpoint 的大小与合同不一致")
    checkpoint = _raw_checkpoint(path)
    if int(checkpoint.get("epoch", -1)) != int(contract["checkpoint_epoch_index"]):
        raise ContractError("Control checkpoint 的内部 epoch 不是 199")
    if int(checkpoint.get("updates", -1)) != int(contract["ema_updates"]):
        raise ContractError("Control checkpoint 的 EMA updates 与合同不一致")
    optimizer = checkpoint.get("optimizer") or {}
    if len(optimizer.get("state") or {}) != int(contract["optimizer_state_count"]):
        raise ContractError("Control checkpoint 的 optimizer state 数量与合同不一致")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "size_bytes": actual_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "completed_epoch": int(checkpoint["epoch"]) + 1,
        "ema_updates": int(checkpoint["updates"]),
        "optimizer_state_count": len(optimizer["state"]),
    }


def _custom_layer_count(model: Any) -> int:
    from src.models.dysample import DySample

    return sum(isinstance(module, DySample) for module in model.modules())


def assert_dysample_architecture(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    from src.models.dysample import DySample
    from ultralytics.utils.torch_utils import unwrap_model

    unwrapped = unwrap_model(model)
    layers = unwrapped.model
    expected_indices = list(config["architecture_contract"]["replacement_layers"])
    actual_indices = [index for index, layer in enumerate(layers) if isinstance(layer, DySample)]
    if actual_indices != expected_indices:
        raise ContractError(
            f"DySample 层位置不符: expected={expected_indices}, actual={actual_indices}"
        )
    layer_reports: list[dict[str, Any]] = []
    for index in expected_indices:
        layer = layers[index]
        expected = config["architecture_contract"]
        if (
            layer.in_channels != 512
            or layer.scale != int(expected["scale"])
            or layer.style != str(expected["style"])
            or layer.groups != int(expected["groups"])
            or hasattr(layer, "scope")
        ):
            raise ContractError(f"第 {index} 层 DySample 属性与实验合同不一致")
        if list(layer.offset.weight.shape) != [32, 512, 1, 1]:
            raise ContractError(f"第 {index} 层 offset.weight 形状异常")
        if list(layer.offset.bias.shape) != [32]:
            raise ContractError(f"第 {index} 层 offset.bias 形状异常")
        layer_reports.append(
            {
                "index": index,
                "in_channels": layer.in_channels,
                "scale": layer.scale,
                "style": layer.style,
                "groups": layer.groups,
                "dyscope": False,
            }
        )
    strides = [float(value) for value in unwrapped.stride.cpu().tolist()]
    if strides != [8.0, 16.0, 32.0]:
        raise ContractError(f"检测头 stride 发生变化: {strides}")
    return {"layers": layer_reports, "stride": strides}


def prepared_checkpoint_report(
    config: dict[str, Any], prepared_path: Path | None = None
) -> dict[str, Any]:
    paths = experiment_paths(config)
    prepared_path = prepared_path or paths["prepared"]
    if not prepared_path.is_file():
        raise FileNotFoundError(f"分支 checkpoint 不存在: {prepared_path}")
    checkpoint = _raw_checkpoint(prepared_path)
    marker = checkpoint.get("e14b_branch") or {}
    if marker.get("format_version") != 1:
        raise ContractError("分支 checkpoint 缺少 E14B v1 标记")
    if str(marker.get("source_sha256", "")).upper() != str(
        config["source_contract"]["sha256"]
    ).upper():
        raise ContractError("分支 checkpoint 的 source 指纹与配置不一致")
    if int(checkpoint.get("epoch", -1)) != int(
        config["source_contract"]["checkpoint_epoch_index"]
    ):
        raise ContractError("分支 checkpoint 的 epoch 元数据不等于 169")
    if checkpoint.get("model") is not None:
        raise ContractError("E14B checkpoint 必须使用 model=None、ema=模型的格式")
    model = checkpoint.get("ema")
    if model is None or _custom_layer_count(model) != 2:
        raise ContractError("分支 checkpoint 必须包含恰好两层 DySample")
    architecture_report = assert_dysample_architecture(model, config)

    optimizer = checkpoint.get("optimizer") or {}
    group_sizes = [len(group.get("params") or []) for group in optimizer.get("param_groups") or []]
    expected_source_groups = list(config["source_contract"]["optimizer_group_sizes"])
    if group_sizes != expected_source_groups:
        raise ContractError(
            f"prepared checkpoint 应暂存 source 分组，expected={expected_source_groups}, actual={group_sizes}"
        )
    source_names = marker.get("source_optimizer_group_names") or []
    if [len(group) for group in source_names] != expected_source_groups:
        raise ContractError("prepared checkpoint 内的 source 参数名称映射损坏")

    target_groups = canonical_musgd_group_names(model)
    target_sizes = [len(group) for group in target_groups]
    expected_target = list(config["architecture_contract"]["expected_target_group_sizes"])
    if target_sizes != expected_target:
        raise ContractError(
            f"DySample 目标分组不符，expected={expected_target}, actual={target_sizes}"
        )
    expected_new = sorted(config["architecture_contract"]["expected_new_parameter_names"])
    source_set = set(_flatten(source_names))
    target_set = set(_flatten(target_groups))
    if sorted(target_set - source_set) != expected_new:
        raise ContractError("prepared checkpoint 新增可训练参数不止预期的四项")

    # Re-derive the source name order and rebuild the seeded target so an old or
    # tampered prepared artifact cannot be accepted merely because counts match.
    from src.models.ultralytics_registry import build_custom_yolo

    source_checkpoint = _raw_checkpoint(paths["source"])
    source_model = source_checkpoint.get("ema") or source_checkpoint.get("model")
    if source_model is None:
        raise ContractError("source checkpoint 缺少模型")
    rederived_source_names = canonical_musgd_group_names(source_model)
    if source_names != rederived_source_names:
        raise ContractError("prepared checkpoint 的 source 参数名称顺序无法由固定 source 重现")
    set_seed(int(config["training"]["seed"]))
    expected_yolo = build_custom_yolo(paths["model_yaml"])
    expected_model = expected_yolo.model.float()
    expected_model.load(source_model.float())
    expected_state = expected_model.half().state_dict()
    actual_state = model.state_dict()
    if set(expected_state) != set(actual_state):
        raise ContractError("prepared checkpoint 的模型 state key 集合无法重现")
    mismatched_values = [
        key for key in expected_state if not expected_state[key].equal(actual_state[key])
    ]
    if mismatched_values:
        raise ContractError(
            f"prepared checkpoint 的模型状态无法由固定输入重现: {mismatched_values[:8]}"
        )
    import torch

    if not all(
        torch.isfinite(value).all()
        for value in actual_state.values()
        if isinstance(value, torch.Tensor)
    ):
        raise ContractError("prepared checkpoint 模型含 NaN/Inf")

    scaler = checkpoint.get("scaler") or {}
    if float(scaler.get("scale", -1.0)) != float(config["source_contract"]["amp_scale"]):
        raise ContractError("AMP scaler scale 未保留")
    if int(scaler.get("_growth_tracker", -1)) != int(
        config["source_contract"]["amp_growth_tracker"]
    ):
        raise ContractError("AMP scaler growth tracker 未保留")
    if int(checkpoint.get("updates", -1)) != int(config["source_contract"]["ema_updates"]):
        raise ContractError("EMA updates 未保留")

    return {
        "status": "ready",
        "prepared_checkpoint": str(prepared_path),
        "prepared_sha256": sha256_file(prepared_path),
        "checkpoint_epoch_index": int(checkpoint["epoch"]),
        "completed_epoch": int(checkpoint["epoch"]) + 1,
        "dy_sample_layers": 2,
        "architecture": architecture_report,
        "shared_model_state_items": int(marker["shared_model_state_items"]),
        "new_model_state_items": marker["new_model_state_items"],
        "source_optimizer_group_sizes": group_sizes,
        "target_optimizer_group_sizes": target_sizes,
        "source_optimizer_state_count": len(optimizer.get("state") or {}),
        "new_trainable_parameter_names": expected_new,
        "ema_updates": int(checkpoint["updates"]),
        "amp_scale": float(scaler["scale"]),
        "amp_growth_tracker": int(scaler["_growth_tracker"]),
        "scheduler_anchor_epoch": int(marker["scheduler"]["anchor_epoch"]),
        "scheduler_horizon_epoch": int(marker["scheduler"]["schedule_horizon_epoch"]),
        "framework_state_strict": True,
        "bitwise_continuation": False,
    }


def prepare_branch_checkpoint(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Create the new E14B checkpoint without writing to the source model."""
    paths = experiment_paths(config)
    static_contract = assert_static_contract(config)
    source_before = assert_source_checkpoint(config)
    control_before = assert_control_checkpoint(config)
    prepared = paths["prepared"]
    if prepared.exists():
        report = prepared_checkpoint_report(config, prepared)
        source_after = assert_source_checkpoint(config)
        control_after = assert_control_checkpoint(config)
        if source_before != source_after:
            raise ContractError("只读 source checkpoint 在审计期间发生变化")
        if control_before != control_after:
            raise ContractError("只读 Control checkpoint 在审计期间发生变化")
        report["static_contract"] = static_contract
        report["source_before"] = source_before
        report["source_after"] = source_after
        report["control_before"] = control_before
        report["control_after"] = control_after
        _atomic_json(paths["artifact_dir"] / "preparation_report.json", report)
        return prepared, report

    from src.models.ultralytics_registry import build_custom_yolo, register_custom_modules
    from ultralytics.nn.tasks import load_checkpoint

    register_custom_modules()
    seed = int(config["training"]["seed"])
    set_seed(seed)
    source_model, source_checkpoint = load_checkpoint(paths["source"], device="cpu")
    if int(source_checkpoint.get("epoch", -1)) != int(
        config["source_contract"]["checkpoint_epoch_index"]
    ):
        raise ContractError("source checkpoint 的 epoch 元数据已变化")
    if source_checkpoint.get("optimizer") is None or source_checkpoint.get("scaler") is None:
        raise ContractError("source checkpoint 缺少 optimizer 或 scaler，不能连续迁移")

    source_groups = canonical_musgd_group_names(source_model)
    expected_source_groups = list(config["source_contract"]["optimizer_group_sizes"])
    if [len(group) for group in source_groups] != expected_source_groups:
        raise ContractError("source 模型的参数分组与 checkpoint 合同不一致")
    source_optimizer = source_checkpoint["optimizer"]
    saved_group_sizes = [len(group["params"]) for group in source_optimizer["param_groups"]]
    if saved_group_sizes != expected_source_groups:
        raise ContractError("source optimizer 参数组与 source 模型不一致")
    if len(source_optimizer.get("state") or {}) != int(
        config["source_contract"]["optimizer_state_count"]
    ):
        raise ContractError("source optimizer state 数量与配置不一致")
    for group in source_optimizer["param_groups"]:
        if not math.isclose(float(group.get("momentum", -1.0)), 0.937, rel_tol=0, abs_tol=1e-12):
            raise ContractError("source optimizer momentum 不是 0.937")

    target_yolo = build_custom_yolo(paths["model_yaml"])
    target_model = target_yolo.model.float()
    target_model.load(source_model)
    target_model.names = deepcopy(getattr(source_model, "names", target_model.names))
    target_model.args = deepcopy(source_checkpoint.get("train_args") or {})
    target_model.task = getattr(source_model, "task", "detect")
    assert_dysample_architecture(target_model, config)

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    shared_keys = [
        key
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    ]
    new_model_state_items = sorted(set(target_state) - set(shared_keys))
    expected_new_model_items = sorted(
        [
            "model.11.init_pos",
            "model.11.offset.weight",
            "model.11.offset.bias",
            "model.14.init_pos",
            "model.14.offset.weight",
            "model.14.offset.bias",
        ]
    )
    if len(shared_keys) != len(source_state):
        missing = sorted(set(source_state) - set(shared_keys))
        raise ContractError(f"共享模型状态未完整迁移: {missing[:8]}")
    if new_model_state_items != expected_new_model_items:
        raise ContractError(
            f"模型新增 state 不符合单变量约束: {new_model_state_items}"
        )
    unequal = [
        key for key in shared_keys if not source_state[key].equal(target_state[key])
    ]
    if unequal:
        raise ContractError(f"共享模型状态值未精确复制: {unequal[:8]}")

    target_groups = canonical_musgd_group_names(target_model)
    expected_target_groups = list(config["architecture_contract"]["expected_target_group_sizes"])
    if [len(group) for group in target_groups] != expected_target_groups:
        raise ContractError("目标模型 MuSGD 分组与配置不一致")
    source_shapes = parameter_shapes(source_model)
    expected_new_parameters = sorted(
        config["architecture_contract"]["expected_new_parameter_names"]
    )
    if sorted(set(_flatten(target_groups)) - set(_flatten(source_groups))) != expected_new_parameters:
        raise ContractError("目标模型新增参数不止四个 DySample offset 参数")

    train_args = deepcopy(source_checkpoint.get("train_args") or {})
    run_dir = paths["run_project"] / config["training"]["run_name"]
    train_args.update(
        {
            "model": str(paths["model_yaml"]),
            "data": str(paths["data_yaml"]),
            "epochs": int(config["training"]["stop_epoch"]),
            "project": str(paths["run_project"]),
            "name": str(config["training"]["run_name"]),
            "save_dir": str(run_dir),
            "optimizer": "MuSGD",
            "warmup_epochs": 0.0,
            "close_mosaic": int(config["training"]["stop_epoch"])
            - int(config["scheduler"]["anchor_epoch"]),
            "val": False,
            "resume": False,
        }
    )
    target_model.args = deepcopy(train_args)

    marker = {
        "format_version": 1,
        "experiment_id": "E14B",
        "created_at": dt.datetime.now().isoformat(),
        "source_checkpoint": str(paths["source"]),
        "source_sha256": source_before["sha256"],
        "source_size_bytes": source_before["size_bytes"],
        "source_checkpoint_epoch_index": int(source_checkpoint["epoch"]),
        "source_optimizer_group_names": source_groups,
        "source_parameter_shapes": source_shapes,
        "expected_new_parameter_names": expected_new_parameters,
        "shared_model_state_items": len(shared_keys),
        "new_model_state_items": new_model_state_items,
        "scheduler": deepcopy(config["scheduler"]),
        "notes": (
            "Framework-state strict architecture migration; RNG/dataloader state is not "
            "serialized by Ultralytics, so this is not bitwise continuation."
        ),
    }
    branch_checkpoint = dict(source_checkpoint)
    branch_checkpoint.update(
        {
            "model": None,
            "ema": deepcopy(target_model).half(),
            "optimizer": deepcopy(source_optimizer),
            "scaler": deepcopy(source_checkpoint["scaler"]),
            "updates": int(source_checkpoint["updates"]),
            "train_args": train_args,
            "e14b_branch": marker,
        }
    )
    _atomic_torch_save(branch_checkpoint, prepared)

    source_after = assert_source_checkpoint(config)
    control_after = assert_control_checkpoint(config)
    if source_before != source_after:
        raise ContractError("只读 source checkpoint 在准备期间发生变化")
    if control_before != control_after:
        raise ContractError("只读 Control checkpoint 在准备期间发生变化")
    report = prepared_checkpoint_report(config, prepared)
    report["static_contract"] = static_contract
    report["source_before"] = source_before
    report["source_after"] = source_after
    report["control_before"] = control_before
    report["control_after"] = control_after
    _atomic_json(paths["artifact_dir"] / "preparation_report.json", report)
    return prepared, report


@dataclass(frozen=True)
class TrainerContext:
    config: dict[str, Any]
    resume_checkpoint: Path
    run_dir: Path
    mode: str
    stop_epoch: int
    batch: int
    workers: int
    fraction: float
    imgsz: int
    save_epochs: tuple[int, ...]


def _build_context(config: dict[str, Any], mode: str, resume_checkpoint: Path) -> TrainerContext:
    paths = experiment_paths(config)
    section = config["smoke_test"] if mode == "smoke" else config["training"]
    run_dir = paths["run_project"] / str(section["run_name"])
    return TrainerContext(
        config=config,
        resume_checkpoint=resume_checkpoint,
        run_dir=run_dir,
        mode=mode,
        stop_epoch=int(section["stop_epoch"]),
        batch=int(section["batch"]),
        workers=int(section["workers"]),
        fraction=float(section.get("fraction", 1.0)),
        imgsz=int(section["imgsz"]),
        save_epochs=tuple(int(value) for value in section["save_epochs"]),
    )


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _expected_lineage(
    config: dict[str, Any], mode: str, run_name: str
) -> dict[str, Any]:
    static = assert_static_contract(config)
    return {
        "format_version": 2,
        "experiment_id": "E14B",
        "mode": mode,
        "run_name": run_name,
        "source_sha256": str(config["source_contract"]["sha256"]).upper(),
        "control_sha256": str(config["control_contract"]["sha256"]).upper(),
        "model_yaml_sha256": static["model_yaml_sha256"],
        "dysample_impl_sha256": static["dysample_impl_sha256"],
        "data_yaml_sha256": static["data_yaml_sha256"],
        "runner_sha256": static["runner_sha256"],
        "ultralytics_version": static["ultralytics_version"],
        "scheduler": deepcopy(config["scheduler"]),
        "architecture": {
            "replacement_layers": list(
                config["architecture_contract"]["replacement_layers"]
            ),
            "style": str(config["architecture_contract"]["style"]),
            "scale": int(config["architecture_contract"]["scale"]),
            "groups": int(config["architecture_contract"]["groups"]),
            "dyscope": False,
        },
    }


def _attach_optimizer_checkpoint_metadata(
    optimizer: Any, model: Any, context: TrainerContext
) -> dict[str, Any]:
    group_names = canonical_musgd_group_names(model)
    if len(group_names) != len(optimizer.param_groups):
        raise ContractError("保存前 optimizer 参数组数与名称组数不一致")
    for group, names in zip(optimizer.param_groups, group_names):
        if len(group["params"]) != len(names):
            raise ContractError("保存前 optimizer 参数组长度与名称组长度不一致")
        group["e14b_param_names"] = list(names)
    metadata = {
        "lineage": _expected_lineage(context.config, context.mode, context.run_dir.name),
        "target_parameter_shapes": parameter_shapes(model),
        "optimizer_group_names_sha256": _json_sha256(group_names),
        "expected_optimizer_state_count": int(
            context.config["source_contract"]["optimizer_state_count"]
        )
        + len(context.config["architecture_contract"]["expected_new_parameter_names"]),
        "only_parameter_without_state": "model.23.dfl.conv.weight",
    }
    optimizer.param_groups[0]["e14b_metadata"] = metadata
    return metadata


def _extract_and_validate_optimizer_metadata(
    optimizer_state: dict[str, Any],
    model: Any,
    config: dict[str, Any],
    mode: str,
    run_name: str,
) -> tuple[list[list[str]], dict[str, Any]]:
    groups = optimizer_state.get("param_groups") or []
    if len(groups) != 8:
        raise ContractError("E14B 自身 checkpoint 必须有 8 个 optimizer 参数组")
    saved_names = [group.get("e14b_param_names") for group in groups]
    if any(not isinstance(names, list) for names in saved_names):
        raise ContractError("E14B 自身 checkpoint 缺少逐组参数名称，拒绝按位置恢复")
    names = [list(group_names) for group_names in saved_names]
    metadata = groups[0].get("e14b_metadata")
    if not isinstance(metadata, dict):
        raise ContractError("E14B 自身 checkpoint 缺少来源与调度元数据")
    expected_lineage = _expected_lineage(config, mode, run_name)
    if metadata.get("lineage") != expected_lineage:
        raise ContractError("E14B checkpoint 的来源/代码/调度合同与当前运行不一致")
    if metadata.get("optimizer_group_names_sha256") != _json_sha256(names):
        raise ContractError("E14B checkpoint 的 optimizer 参数名称摘要损坏")

    canonical_names = canonical_musgd_group_names(model)
    if names != canonical_names:
        raise ContractError("E14B checkpoint 的参数名称顺序与当前目标架构不一致")
    if metadata.get("target_parameter_shapes") != parameter_shapes(model):
        raise ContractError("E14B checkpoint 的参数形状合同与当前模型不一致")
    return names, metadata


def _checkpoint_summary(
    path: Path,
    config: dict[str, Any],
    expected_mode: str | None = None,
    expected_run_name: str | None = None,
) -> dict[str, Any]:
    checkpoint = _raw_checkpoint(path)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None or _custom_layer_count(model) != 2:
        raise ContractError(f"恢复目标不是 E14B DySample checkpoint: {path}")
    architecture_report = assert_dysample_architecture(model, config)
    if checkpoint.get("optimizer") is None or checkpoint.get("scaler") is None:
        raise ContractError(f"恢复目标缺少 optimizer/scaler: {path}")
    optimizer = checkpoint["optimizer"]
    group_sizes = [len(group["params"]) for group in optimizer["param_groups"]]
    epoch_index = int(checkpoint.get("epoch", -1))
    if epoch_index < int(config["source_contract"]["checkpoint_epoch_index"]):
        raise ContractError(f"恢复目标早于 epoch170: {path}")
    marker = bool(checkpoint.get("e14b_branch"))
    expected_sizes = (
        list(config["source_contract"]["optimizer_group_sizes"])
        if marker
        else list(config["architecture_contract"]["expected_target_group_sizes"])
    )
    if group_sizes != expected_sizes:
        raise ContractError(
            f"checkpoint optimizer 分组异常: expected={expected_sizes}, actual={group_sizes}"
        )
    native_resume_audit: dict[str, Any] | None = None
    native_mapping_report: dict[str, Any] | None = None
    if not marker:
        if expected_mode is None or expected_run_name is None:
            raise ContractError("审计 E14B 自身 checkpoint 时必须指定 mode 和 run_name")
        saved_names, metadata = _extract_and_validate_optimizer_metadata(
            optimizer,
            model,
            config,
            expected_mode,
            expected_run_name,
        )
        expected_state_count = int(metadata["expected_optimizer_state_count"])
        if len(optimizer.get("state") or {}) != expected_state_count:
            raise ContractError(
                f"E14B 自身 checkpoint state 不完整: "
                f"expected={expected_state_count}, actual={len(optimizer.get('state') or {})}"
            )

        # Prove that a checkpoint written by this experiment can be remapped by
        # its persisted names into a fresh canonical target optimizer.
        from ultralytics.optim import MuSGD

        model = model.float()
        group_names = canonical_musgd_group_names(model)
        named_parameters = dict(model.named_parameters())
        optimizer_groups: list[dict[str, Any]] = []
        for saved_group, names in zip(optimizer["param_groups"], group_names):
            group = deepcopy(saved_group)
            group["params"] = [named_parameters[name] for name in names]
            optimizer_groups.append(group)
        fresh_optimizer = MuSGD(optimizer_groups, muon=0.2, sgd=1.0)
        mapped, native_mapping_report = remap_optimizer_state_dict(
            optimizer,
            saved_names,
            metadata["target_parameter_shapes"],
            fresh_optimizer,
            group_names,
            model,
            [],
        )
        fresh_optimizer.load_state_dict(mapped)
        native_resume_audit = _state_tensor_audit(fresh_optimizer)
        missing_state = _optimizer_parameters_without_state(fresh_optimizer, model)
        if missing_state != [metadata["only_parameter_without_state"]]:
            raise ContractError(
                f"E14B 自身 checkpoint 无状态参数集合异常: {missing_state}"
            )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "checkpoint_epoch_index": epoch_index,
        "completed_epoch": epoch_index + 1,
        "optimizer_group_sizes": group_sizes,
        "optimizer_state_count": len(optimizer.get("state") or {}),
        "ema_updates": int(checkpoint.get("updates", -1)),
        "has_preparation_marker": marker,
        "architecture": architecture_report,
        "native_optimizer_reload_audit": native_resume_audit,
        "native_optimizer_mapping_report": native_mapping_report,
    }


def _select_resume_checkpoint(
    config: dict[str, Any], context: TrainerContext, prepared: Path
) -> tuple[Path, dict[str, Any], bool]:
    last = context.run_dir / "weights" / "last.pt"
    if last.exists():
        summary = _checkpoint_summary(
            last,
            config,
            expected_mode=context.mode,
            expected_run_name=context.run_dir.name,
        )
        if summary["completed_epoch"] > context.stop_epoch:
            raise ContractError(
                f"实验 checkpoint 已超过目标 epoch{context.stop_epoch}: {last}"
            )
        if summary["completed_epoch"] in context.save_epochs:
            selected = context.run_dir / "weights" / f"epoch{summary['completed_epoch']}.pt"
            if not selected.exists():
                shutil.copy2(last, selected)
                summary["recovered_selected_checkpoint"] = str(selected)
        return last, summary, summary["completed_epoch"] == context.stop_epoch
    if context.run_dir.exists() and any(context.run_dir.iterdir()):
        raise ContractError(
            f"实验目录非空但没有可恢复的 last.pt，拒绝静默重建状态: {context.run_dir}"
        )
    summary = _checkpoint_summary(prepared, config)
    return prepared, summary, False


def _state_tensor_audit(optimizer: Any) -> dict[str, Any]:
    import torch

    tensor_count = 0
    nonfinite: list[str] = []
    shape_mismatch: list[str] = []
    schema_mismatch: list[str] = []
    group_by_parameter_id = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for parameter, state in optimizer.state.items():
        expected_keys = {"momentum_buffer"}
        if group_by_parameter_id[id(parameter)].get("use_muon", False):
            expected_keys.add("momentum_buffer_SGD")
        if set(state) != expected_keys:
            schema_mismatch.append(
                f"expected={sorted(expected_keys)}, actual={sorted(state)}"
            )
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            tensor_count += 1
            if not torch.isfinite(value).all():
                nonfinite.append(key)
            if value.ndim > 0 and tuple(value.shape) != tuple(parameter.shape):
                shape_mismatch.append(key)
    if nonfinite or shape_mismatch or schema_mismatch:
        raise ContractError(
            "优化器状态审计失败: "
            f"nonfinite={nonfinite[:5]}, shape_mismatch={shape_mismatch[:5]}, "
            f"schema_mismatch={schema_mismatch[:5]}"
        )
    return {
        "optimizer_state_count": len(optimizer.state),
        "optimizer_state_tensor_count": tensor_count,
        "all_state_tensors_finite": True,
        "all_non_scalar_state_shapes_match": True,
        "all_state_schemas_match_musgd_groups": True,
    }


def _optimizer_parameters_without_state(optimizer: Any, model: Any) -> list[str]:
    from ultralytics.utils.torch_utils import unwrap_model

    name_by_id = {
        id(parameter): name
        for name, parameter in unwrap_model(model).named_parameters()
    }
    state_ids = {id(parameter) for parameter in optimizer.state}
    grouped_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    return sorted(name_by_id[parameter_id] for parameter_id in grouped_ids - state_ids)


def make_trainer_class(context: TrainerContext) -> type[Any]:
    """Build an isolated trainer class bound to one immutable run context."""
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.optim import MuSGD
    from ultralytics.utils.torch_utils import unwrap_model

    config = context.config
    schedule = config["scheduler"]
    expected_target_groups = list(config["architecture_contract"]["expected_target_group_sizes"])
    expected_new_names = tuple(config["architecture_contract"]["expected_new_parameter_names"])

    class E14BStateMigrationTrainer(DetectionTrainer):
        """Restore E13 state into the DySample architecture with hard invariants."""

        def check_resume(self, overrides):  # type: ignore[no-untyped-def]
            super().check_resume(overrides)
            if not self.resume:
                raise ContractError("E14B 只允许从已审计 checkpoint 恢复，不能 fresh train")
            args = self.args
            args.epochs = context.stop_epoch
            args.data = str(experiment_paths(config)["data_yaml"])
            args.project = str(experiment_paths(config)["run_project"])
            args.name = context.run_dir.name
            args.save_dir = str(context.run_dir)
            args.exist_ok = True
            args.imgsz = context.imgsz
            args.batch = context.batch
            args.workers = context.workers
            args.device = str(config["training"]["device"])
            args.seed = int(config["training"]["seed"])
            args.deterministic = True
            args.optimizer = "MuSGD"
            args.warmup_epochs = 0.0
            args.val = False
            args.plots = False
            args.compile = False
            args.save = True
            args.save_period = -1
            args.fraction = context.fraction
            args.time = None
            args.close_mosaic = context.stop_epoch - int(schedule["anchor_epoch"])

        def build_optimizer(  # type: ignore[no-untyped-def]
            self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5
        ):
            optimizer = super().build_optimizer(
                model=model,
                name="MuSGD",
                lr=lr,
                momentum=momentum,
                decay=decay,
                iterations=iterations,
            )
            if not isinstance(optimizer, MuSGD):
                raise ContractError(f"目标 optimizer 不是 MuSGD: {type(optimizer).__name__}")
            if not math.isclose(float(optimizer.muon), 0.2, rel_tol=0, abs_tol=1e-12):
                raise ContractError("MuSGD muon 系数不是 0.2")
            if not math.isclose(float(optimizer.sgd), 1.0, rel_tol=0, abs_tol=1e-12):
                raise ContractError("MuSGD sgd 系数不是 1.0")
            names = canonical_musgd_group_names(model)
            sizes = [len(group) for group in names]
            if sizes != expected_target_groups:
                raise ContractError(
                    f"目标 optimizer 分组异常: expected={expected_target_groups}, actual={sizes}"
                )
            self._e14b_target_group_names = names
            return optimizer

        def _build_train_pipeline(self):
            build_count = getattr(self, "_e14b_pipeline_build_count", 0)
            if build_count:
                raise RuntimeError(
                    "E14B 首轮发生显存/后端错误。为防止自动重建 optimizer 丢失迁移状态，"
                    "本实验已硬停止；请先降低配置 batch 并重新做独立实验。"
                )
            self._e14b_pipeline_build_count = build_count + 1
            return super()._build_train_pipeline()

        def _load_checkpoint_state(self, checkpoint):  # type: ignore[no-untyped-def]
            if checkpoint.get("optimizer") is None or checkpoint.get("scaler") is None:
                raise ContractError("恢复 checkpoint 缺少 optimizer/scaler")
            marker = checkpoint.get("e14b_branch")
            migration_report: dict[str, Any]
            checkpoint_to_load = checkpoint
            if marker:
                if marker.get("format_version") != 1:
                    raise ContractError("不支持的 E14B 分支 checkpoint 版本")
                target_names = canonical_musgd_group_names(self.model)
                mapped_optimizer, migration_report = remap_optimizer_state_dict(
                    checkpoint["optimizer"],
                    marker["source_optimizer_group_names"],
                    marker["source_parameter_shapes"],
                    self.optimizer,
                    target_names,
                    self.model,
                    marker["expected_new_parameter_names"],
                )
                checkpoint_to_load = dict(checkpoint)
                checkpoint_to_load["optimizer"] = mapped_optimizer
                migration_report["restore_kind"] = "source_to_dysample_name_migration"
            else:
                source_names, metadata = _extract_and_validate_optimizer_metadata(
                    checkpoint["optimizer"],
                    self.model,
                    config,
                    context.mode,
                    context.run_dir.name,
                )
                target_names = canonical_musgd_group_names(self.model)
                mapped_optimizer, migration_report = remap_optimizer_state_dict(
                    checkpoint["optimizer"],
                    source_names,
                    metadata["target_parameter_shapes"],
                    self.optimizer,
                    target_names,
                    self.model,
                    [],
                )
                if len(checkpoint["optimizer"].get("state") or {}) != int(
                    metadata["expected_optimizer_state_count"]
                ):
                    raise ContractError("E14B 自身 checkpoint 的 optimizer state 不完整")
                checkpoint_to_load = dict(checkpoint)
                checkpoint_to_load["optimizer"] = mapped_optimizer
                migration_report["restore_kind"] = "native_e14b_name_migration"

            super()._load_checkpoint_state(checkpoint_to_load)
            if not isinstance(self.optimizer, MuSGD):
                raise ContractError("恢复后 optimizer 类型发生变化")
            assert_dysample_architecture(self.model, config)
            if int(self.ema.updates) != int(checkpoint.get("updates", -1)):
                raise ContractError("EMA update 计数恢复失败")
            if not self.amp or not self.scaler.is_enabled():
                raise ContractError("E14B 要求 CUDA AMP 与 GradScaler 均处于启用状态")
            if self.scaler.state_dict() != checkpoint["scaler"]:
                raise ContractError("AMP scaler 状态未被逐字段恢复")

            audit = _state_tensor_audit(self.optimizer)
            migration_report.update(audit)
            if marker:
                if audit["optimizer_state_count"] != int(
                    config["source_contract"]["optimizer_state_count"]
                ):
                    raise ContractError("迁移后的 optimizer state 数量不是 330")
                named_parameters = dict(unwrap_model(self.model).named_parameters())
                state_parameter_ids = {id(parameter) for parameter in self.optimizer.state}
                initialized_new = [
                    name
                    for name in expected_new_names
                    if id(named_parameters[name]) in state_parameter_ids
                ]
                if initialized_new:
                    raise ContractError(
                        f"新增参数不应预置历史 optimizer state: {initialized_new}"
                    )
                expected_without_state = sorted(
                    ["model.23.dfl.conv.weight", *expected_new_names]
                )
            else:
                if audit["optimizer_state_count"] != int(
                    metadata["expected_optimizer_state_count"]
                ):
                    raise ContractError("E14B 自身恢复后的 optimizer state 数量异常")
                expected_without_state = [metadata["only_parameter_without_state"]]
            actual_without_state = _optimizer_parameters_without_state(
                self.optimizer, self.model
            )
            if actual_without_state != expected_without_state:
                raise ContractError(
                    f"恢复后无 optimizer state 的参数集合异常: {actual_without_state}"
                )
            migration_report["parameters_without_state"] = actual_without_state
            self._e14b_restore_report = migration_report

        def resume_training(self, checkpoint):  # type: ignore[no-untyped-def]
            super().resume_training(checkpoint)
            if self.start_epoch <= int(config["source_contract"]["checkpoint_epoch_index"]):
                raise ContractError("E14B 恢复起点没有越过已完成的 epoch170")
            if self.start_epoch >= context.stop_epoch:
                raise ContractError("恢复 checkpoint 已到达或超过本次停止点")

            loaded_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            for group, current_lr in zip(self.optimizer.param_groups, loaded_lrs):
                group.setdefault("initial_lr", current_lr)

            def lr_factor(epoch: int) -> float:
                return continuation_lr_factor(
                    epoch,
                    start_epoch=int(schedule["anchor_epoch"]),
                    schedule_horizon_epoch=int(schedule["schedule_horizon_epoch"]),
                    final_factor=float(schedule["final_lr_factor"]),
                )

            checkpoint_epoch = int(checkpoint["epoch"])
            expected_current_lrs = [
                float(group["initial_lr"]) * lr_factor(checkpoint_epoch)
                for group in self.optimizer.param_groups
            ]
            for index, (actual, expected) in enumerate(
                zip(loaded_lrs, expected_current_lrs)
            ):
                if not math.isclose(actual, expected, rel_tol=5e-8, abs_tol=1e-12):
                    raise ContractError(
                        f"第 {index} 组 checkpoint LR 不在 E13 曲线上: "
                        f"actual={actual:.16g}, expected={expected:.16g}"
                    )

            self.lf = lr_factor
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=self.lf
            )
            for group, current_lr in zip(self.optimizer.param_groups, loaded_lrs):
                group["lr"] = current_lr
            self.scheduler.last_epoch = self.start_epoch - 1
            self.scheduler._last_lr = loaded_lrs
            self._close_dataloader_mosaic()

            next_lrs = [
                float(group["initial_lr"]) * lr_factor(self.start_epoch)
                for group in self.optimizer.param_groups
            ]
            report = dict(getattr(self, "_e14b_restore_report", {}))
            report.update(
                {
                    "checkpoint": str(context.resume_checkpoint),
                    "checkpoint_epoch_index": checkpoint_epoch,
                    "completed_epoch": checkpoint_epoch + 1,
                    "next_epoch": self.start_epoch + 1,
                    "stop_epoch": context.stop_epoch,
                    "loaded_lrs": loaded_lrs,
                    "expected_next_lrs": next_lrs,
                    "scheduler_last_epoch": self.scheduler.last_epoch,
                    "scheduler_anchor_epoch": int(schedule["anchor_epoch"]),
                    "scheduler_horizon_epoch": int(schedule["schedule_horizon_epoch"]),
                    "mosaic_closed": True,
                    "ema_updates": int(self.ema.updates),
                    "amp_enabled": bool(self.amp),
                    "scaler_scale": float(self.scaler.get_scale()),
                }
            )
            self._e14b_restore_report = report
            _atomic_json(context.run_dir / "state_restore_report.json", report)

        def validate(self):  # type: ignore[no-untyped-def]
            return {}, self.fitness

        def final_eval(self) -> None:
            # Avoid both train-backed validation and optimizer stripping.  The
            # latter is required so the final epoch200 checkpoint stays resumable.
            return None

        def save_model(self) -> bool:
            named_parameters = dict(unwrap_model(self.model).named_parameters())
            state_parameter_ids = {id(parameter) for parameter in self.optimizer.state}
            missing_new_state = [
                name
                for name in expected_new_names
                if id(named_parameters[name]) not in state_parameter_ids
            ]
            if missing_new_state:
                raise ContractError(
                    f"DySample 新参数在训练后仍无 optimizer state: {missing_new_state}"
                )
            state_audit = _state_tensor_audit(self.optimizer)
            expected_state_count = int(config["source_contract"]["optimizer_state_count"]) + len(
                expected_new_names
            )
            if state_audit["optimizer_state_count"] != expected_state_count:
                raise ContractError(
                    f"训练后 optimizer state 应为 {expected_state_count}，"
                    f"实际 {state_audit['optimizer_state_count']}"
                )

            parameters_without_state = _optimizer_parameters_without_state(
                self.optimizer, self.model
            )
            if parameters_without_state != ["model.23.dfl.conv.weight"]:
                raise ContractError(
                    f"训练后无 optimizer state 的参数集合异常: {parameters_without_state}"
                )
            checkpoint_metadata = _attach_optimizer_checkpoint_metadata(
                self.optimizer, self.model, context
            )

            saved = super().save_model()
            epoch_number = self.epoch + 1
            if saved and epoch_number in context.save_epochs:
                selected = self.wdir / f"epoch{epoch_number}.pt"
                if selected.exists():
                    raise FileExistsError(f"拒绝覆盖已有 E14B epoch checkpoint: {selected}")
                shutil.copy2(self.last, selected)
            audit_report = {
                "epoch": epoch_number,
                "loss_finite": bool(torch.isfinite(self.loss).all()),
                "ema_updates": int(self.ema.updates),
                "optimizer": type(self.optimizer).__name__,
                "parameters_without_state": parameters_without_state,
                "checkpoint_lineage": checkpoint_metadata["lineage"],
                **state_audit,
            }
            _atomic_json(context.run_dir / "latest_state_audit.json", audit_report)
            return saved

    return E14BStateMigrationTrainer


@contextlib.contextmanager
def _exclusive_run_lock(config: dict[str, Any], run_name: str):
    lock = experiment_paths(config)["artifact_dir"] / "locks" / f"{run_name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = lock.read_text(encoding="utf-8", errors="replace")
        raise ContractError(
            f"同一实验已有运行锁，拒绝并发写 checkpoint: {lock}\n{owner}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "created_at": dt.datetime.now().isoformat(),
                        "run_name": run_name,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        yield lock
    finally:
        if lock.exists():
            lock.unlink()


def _run_training_locked(
    config: dict[str, Any], mode: str, prepared: Path
) -> dict[str, Any]:
    if mode not in {"smoke", "train"}:
        raise ValueError(f"无效训练模式: {mode}")
    context = _build_context(config, mode, prepared)
    resume_checkpoint, resume_summary, complete = _select_resume_checkpoint(
        config, context, prepared
    )
    if complete:
        selected = context.run_dir / "weights" / f"epoch{context.stop_epoch}.pt"
        if not selected.is_file():
            raise ContractError(f"last.pt 已完成但缺少停止点快照: {selected}")
        result = {
            "status": "already_complete",
            "mode": mode,
            "run_dir": str(context.run_dir),
            "checkpoint": resume_summary,
        }
        _atomic_json(context.run_dir / "completion_report.json", result)
        return result

    context = TrainerContext(
        **{**context.__dict__, "resume_checkpoint": resume_checkpoint}
    )
    from src.models.ultralytics_registry import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    set_seed(int(config["training"]["seed"]))
    trainer_class = make_trainer_class(context)
    model = YOLO(str(resume_checkpoint))
    started_at = dt.datetime.now()
    model.train(
        trainer=trainer_class,
        resume=str(resume_checkpoint),
        device=str(config["training"]["device"]),
        batch=context.batch,
        workers=context.workers,
        imgsz=context.imgsz,
        fraction=context.fraction,
        val=False,
        plots=False,
        compile=False,
        save=True,
        save_period=-1,
    )
    finished_at = dt.datetime.now()

    last = context.run_dir / "weights" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(f"训练结束但没有生成 last.pt: {last}")
    final_summary = _checkpoint_summary(
        last,
        config,
        expected_mode=context.mode,
        expected_run_name=context.run_dir.name,
    )
    if int(final_summary["completed_epoch"]) != context.stop_epoch:
        raise ContractError(
            f"训练未到预定停止点 epoch{context.stop_epoch}: {final_summary}"
        )
    selected = context.run_dir / "weights" / f"epoch{context.stop_epoch}.pt"
    if not selected.is_file():
        raise FileNotFoundError(f"缺少精确保留的停止点 checkpoint: {selected}")
    source_after = assert_source_checkpoint(config)
    control_after = assert_control_checkpoint(config)
    result = {
        "status": "complete",
        "mode": mode,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "run_dir": str(context.run_dir),
        "resume_checkpoint": resume_summary,
        "final_checkpoint": final_summary,
        "selected_checkpoint": str(selected),
        "source_checkpoint_after": source_after,
        "control_checkpoint_after": control_after,
    }
    _atomic_json(context.run_dir / "completion_report.json", result)
    return result


def run_training(config: dict[str, Any], mode: str, prepared: Path) -> dict[str, Any]:
    if mode not in {"smoke", "train"}:
        raise ValueError(f"无效训练模式: {mode}")
    section = config["smoke_test"] if mode == "smoke" else config["training"]
    source_before = assert_source_checkpoint(config)
    control_before = assert_control_checkpoint(config)
    with _exclusive_run_lock(config, str(section["run_name"])):
        result = _run_training_locked(config, mode, prepared)
    source_after = assert_source_checkpoint(config)
    control_after = assert_control_checkpoint(config)
    if source_before != source_after:
        raise ContractError("只读 source checkpoint 在训练期间发生变化")
    if control_before != control_after:
        raise ContractError("只读 Control checkpoint 在训练期间发生变化")
    result["source_checkpoint_before"] = source_before
    result["source_checkpoint_after"] = source_after
    result["control_checkpoint_before"] = control_before
    result["control_checkpoint_after"] = control_after
    _atomic_json(
        (experiment_paths(config)["run_project"] / str(section["run_name"]))
        / "completion_report.json",
        result,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = selected_mode(args)
    config = load_experiment_config(args.config)
    prepared, preparation = prepare_branch_checkpoint(config)
    if mode == "prepare":
        result = {
            "status": "prepared",
            "message": "分支 checkpoint 已准备并审计；未启动训练。",
            "preparation": preparation,
        }
    else:
        result = run_training(config, mode, prepared)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
