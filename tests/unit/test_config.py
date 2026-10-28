"""The shipped recorder configuration loads with the documented types and hardware values."""

from __future__ import annotations

from pathlib import Path

import pytest

from mvsr.calib.handeye import REFERENCE_HANDEYE_SHA256
from mvsr.record.config import ConfigError, load_record_config


def test_shipped_config_types_and_values() -> None:
    cfg = load_record_config()
    wrist, head = cfg.cameras["wrist"], cfg.cameras["head"]
    assert (wrist.serial, wrist.width, wrist.height, wrist.fps) == ("323622271240", 1280, 720, 30)
    assert wrist.depth_units_m == 1e-4 and head.depth_units_m == 1e-3
    assert head.serial == "244622070961" and head.fps == 15 and head.color_format == "rgb8"
    assert all(isinstance(v, float) for v in cfg.disk.est_bytes_per_frame.values())
    assert isinstance(cfg.disk.reserve_gb, float) and isinstance(cfg.robot.poll_hz, float)
    assert cfg.robot.ip == "192.168.1.19" and cfg.robot.source == "poll"
    assert cfg.robot.poll_retries == 3 and cfg.robot.poll_retry_delay_s == pytest.approx(0.02)
    assert cfg.handeye.sha256 == REFERENCE_HANDEYE_SHA256 and cfg.handeye_path().is_file()


def test_config_rejects_string_number(tmp_path: Path) -> None:
    src = (Path(__file__).resolve().parents[2] / "configs/record.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(src.replace("wrist: 3.5e+6", "wrist: 3.5e6"))
    with pytest.raises(ConfigError, match=r"est_bytes_per_frame\.wrist"):
        load_record_config(bad)
    bad.write_text(src.replace("poll_hz: 100", "poll_hz: fast"))
    with pytest.raises(ConfigError, match="poll_hz"):
        load_record_config(bad)
