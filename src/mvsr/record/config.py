"""Recorder configuration (`configs/record.yaml`), validated into frozen dataclasses."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from mvsr.paths import repo_root
from mvsr.yaml_config import ConfigError
from mvsr.yaml_config import load_section as _section

__all__ = ["ConfigError", "RecordConfig", "load_record_config"]

DEFAULT_CONFIG = "configs/record.yaml"


@dataclass(frozen=True)
class CameraConfig:
    serial: str
    width: int
    height: int
    fps: int
    color_format: str
    depth_units_m: float
    writer_threads: int


@dataclass(frozen=True)
class CaptureConfig:
    warmup_frames: int
    first_frame_timeout_ms: int
    frame_timeout_ms: int
    max_consecutive_timeouts: int
    process_queue_size: int
    writer_queue_size: int
    png_compression: int
    fsync_images: bool


@dataclass(frozen=True)
class RobotConfig:
    ip: str
    port: int
    source: str
    poll_hz: float
    udp_port: int
    udp_cycle_5ms_units: int
    udp_host_ip: str | None
    abort_after_failure_s: float
    poll_retries: int
    poll_retry_delay_s: float


@dataclass(frozen=True)
class HandEyeConfig:
    path: str
    sha256: str


@dataclass(frozen=True)
class DiskConfig:
    reserve_gb: float
    est_bytes_per_frame: dict[str, float]
    check_interval_s: float


@dataclass(frozen=True)
class MonitorConfig:
    status_interval_s: float
    preview_max_width: int


@dataclass(frozen=True)
class RecordConfig:
    cameras: dict[str, CameraConfig]
    capture: CaptureConfig
    robot: RobotConfig
    handeye: HandEyeConfig
    disk: DiskConfig
    monitor: MonitorConfig
    source_path: str
    source_sha256: str

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    def handeye_path(self) -> Path:
        p = Path(self.handeye.path)
        return p if p.is_absolute() else repo_root() / p


def load_record_config(path: str | Path | None = None) -> RecordConfig:
    p = Path(path) if path is not None else repo_root() / DEFAULT_CONFIG
    text = p.read_bytes()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: not a mapping")
    cams_raw = raw.get("cameras")
    if not isinstance(cams_raw, dict) or "wrist" not in cams_raw:
        raise ConfigError("cameras.wrist is required")
    cameras = {cid: _section(cams_raw, CameraConfig, cid) for cid in cams_raw}
    cfg = RecordConfig(
        cameras=cameras,
        capture=_section(raw, CaptureConfig, "capture"),
        robot=_section(raw, RobotConfig, "robot"),
        handeye=_section(raw, HandEyeConfig, "handeye"),
        disk=_section(raw, DiskConfig, "disk"),
        monitor=_section(raw, MonitorConfig, "monitor"),
        source_path=str(p),
        source_sha256=hashlib.sha256(text).hexdigest(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: RecordConfig) -> None:
    for cid, cam in cfg.cameras.items():
        if cid not in ("wrist", "head"):
            raise ConfigError(f"unknown camera {cid}")
        if cam.color_format not in ("bgr8", "rgb8"):
            raise ConfigError(f"{cid}.color_format must be bgr8|rgb8")
        if min(cam.width, cam.height, cam.fps, cam.writer_threads) <= 0 or cam.depth_units_m <= 0:
            raise ConfigError(f"{cid}: width/height/fps/writer_threads/depth_units_m must be > 0")
        if cid not in cfg.disk.est_bytes_per_frame:
            raise ConfigError(f"disk.est_bytes_per_frame.{cid} missing")
    c = cfg.capture
    if min(c.process_queue_size, c.writer_queue_size, c.max_consecutive_timeouts) < 1:
        raise ConfigError("capture queue sizes and max_consecutive_timeouts must be >= 1")
    if not 0 <= c.png_compression <= 9:
        raise ConfigError("png_compression must be 0..9")
    if cfg.robot.source not in ("poll", "udp"):
        raise ConfigError("robot.source must be poll|udp")
    if cfg.robot.poll_hz <= 0 or cfg.robot.abort_after_failure_s <= 0:
        raise ConfigError("robot.poll_hz and abort_after_failure_s must be > 0")
    if cfg.robot.poll_retries < 0 or cfg.robot.poll_retry_delay_s < 0:
        raise ConfigError("robot.poll_retries and poll_retry_delay_s must be >= 0")
    if cfg.disk.reserve_gb < 0 or cfg.disk.check_interval_s <= 0:
        raise ConfigError("disk.reserve_gb >= 0 and check_interval_s > 0 required")
