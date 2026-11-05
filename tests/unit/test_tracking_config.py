from __future__ import annotations

from pathlib import Path

import pytest

from mvsr.tracking.config import load_tracking_config
from mvsr.yaml_config import ConfigError


def test_shipped_tracking_config() -> None:
    cfg = load_tracking_config()
    assert cfg.hypothesis.min_inliers == 6
    assert len(cfg.icp.iterations) == len(cfg.icp.max_correspondence_m) == 3
    assert cfg.keyframes.max_keyframes >= cfg.hypothesis.max_local_keyframes


def test_tracking_config_rejects_inconsistent_icp(tmp_path: Path) -> None:
    src = (Path(__file__).resolve().parents[2] / "configs/tracking.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(src.replace("iterations: [3, 2, 1]", "iterations: [8, 6]"))
    with pytest.raises(ConfigError, match="icp"):
        load_tracking_config(bad)
