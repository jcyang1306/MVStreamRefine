from __future__ import annotations

from pathlib import Path

import pytest

from mvsr.keyframes.config import load_keyframe_config
from mvsr.yaml_config import ConfigError


def test_shipped_keyframe_config() -> None:
    cfg = load_keyframe_config()
    assert cfg.retrieval.max_local_keyframes <= cfg.retrieval.max_global_keyframes
    assert cfg.selection.feature_grid_size == 4


def test_keyframe_config_rejects_bad_bounds(tmp_path: Path) -> None:
    src = (Path(__file__).resolve().parents[2] / "configs/keyframes.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(src.replace("max_local_keyframes: 8", "max_local_keyframes: 100"))
    with pytest.raises(ConfigError, match="retrieval"):
        load_keyframe_config(bad)
