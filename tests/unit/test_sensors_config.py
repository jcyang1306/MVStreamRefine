"""The shipped P1 configuration loads with typed values and rejects inconsistent boards."""

from __future__ import annotations

from pathlib import Path

import pytest

from mvsr.sensors.config import load_sensors_config
from mvsr.yaml_config import ConfigError


def test_shipped_sensors_config() -> None:
    cfg = load_sensors_config()
    assert cfg.clock.device_counter_period_us == 1 << 32
    assert cfg.sync.max_interp_gap_ns == 31_000_000
    assert cfg.rgbd.max_color_depth_dt_ms == {"wrist": 3.1, "head": 1.0}
    assert cfg.replay.handeye is not None
    assert (cfg.board.inner_cols, cfg.board.inner_rows, cfg.board.square_mm) == (11, 8, None)
    assert isinstance(cfg.gate.max_board_spread_p95_mm, float)
    assert cfg.snapshot()["board"]["inner_cols"] == 11


def test_sensors_config_rejects_bad_values(tmp_path: Path) -> None:
    src = (Path(__file__).resolve().parents[2] / "configs/sensors.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(src.replace("inner_rows: 8", "inner_rows: 9"))
    with pytest.raises(ConfigError, match="odd"):
        load_sensors_config(bad)
    bad.write_text(src.replace("max_interp_gap_ns: 31000000", "max_interp_gap_ns: 25ms"))
    with pytest.raises(ConfigError, match="max_interp_gap_ns"):
        load_sensors_config(bad)
    bad.write_text(
        src.replace(
            "handeye_sha256: 2b60289b9b68c62b08d40eddc0a06232249c152d12ec1687639e6c2d974d0ef3",
            "handeye_sha256: null",
        )
    )
    with pytest.raises(ConfigError, match=r"replay\.handeye"):
        load_sensors_config(bad)
