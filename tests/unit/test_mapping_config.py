from __future__ import annotations

from pathlib import Path

import pytest

from mvsr.mapping.config import load_mapping_config
from mvsr.yaml_config import ConfigError


def test_shipped_mapping_config() -> None:
    cfg = load_mapping_config()
    assert cfg.tsdf.sdf_trunc_m >= 2 * cfg.tsdf.voxel_length_m
    assert cfg.gate.min_geometry_points >= cfg.output.min_point_count


def test_mapping_config_rejects_small_truncation(tmp_path: Path) -> None:
    src = (Path(__file__).resolve().parents[2] / "configs/mapping.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(src.replace("sdf_trunc_m: 0.015", "sdf_trunc_m: 0.001"))
    with pytest.raises(ConfigError, match="tsdf"):
        load_mapping_config(bad)
