from __future__ import annotations

from pathlib import Path

from common import DEFAULT_CONFIG, activate_version, load_experiment, validate_inputs, version_root


def test_central_config_and_inputs_exist() -> None:
    _, config = load_experiment(DEFAULT_CONFIG)
    data, pretrained, model_yaml = validate_inputs(config)
    assert data.is_file()
    assert pretrained.is_file()
    assert model_yaml.is_file()


def test_all_official_yolo11_scales_are_snapshotted() -> None:
    root = version_root("v001_baseline")
    for scale in "nsm lx".replace(" ", ""):
        assert (root / "models" / f"yolo11{scale}.yaml").is_file()


def test_selected_source_is_local_snapshot() -> None:
    _, config = load_experiment(DEFAULT_CONFIG)
    ultralytics = activate_version("v001_baseline", "8.4.98")
    loaded = Path(ultralytics.__file__).resolve()
    assert version_root("v001_baseline") in loaded.parents
    assert config["version"] == "v001_baseline"

