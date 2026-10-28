"""Recorder configuration (`configs/record.yaml`), validated into frozen dataclasses."""

from __future__ import annotations

import hashlib
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from mvsr.paths import repo_root

DEFAULT_CONFIG = "configs/record.yaml"


class ConfigError(ValueError):
    pass


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


def _typed(value: Any, tp: Any, where: str) -> Any:
    """Check one YAML value against its annotation (YAML 1.1 reads e.g. `3.5e6` as a string)."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        if value is None and type(None) in get_args(tp):
            return None
        (inner,) = [a for a in get_args(tp) if a is not type(None)]
        return _typed(value, inner, where)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        kt, vt = get_args(tp)
        return {_typed(k, kt, where): _typed(v, vt, f"{where}.{k}") for k, v in value.items()}
    if tp is bool:
        ok = isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif tp is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif tp is str:
        ok = isinstance(value, str)
    else:
        raise ConfigError(f"{where}: unsupported annotation {tp!r}")
    if not ok:
        raise ConfigError(f"{where}: expected {getattr(tp, '__name__', tp)}, got {value!r}")
    return value


def _section(raw: dict[str, Any], cls: type[Any], name: str) -> Any:
    data = raw.get(name)
    if not isinstance(data, dict):
        raise ConfigError(f"missing section '{name}'")
    names = set(cls.__dataclass_fields__)
    unknown, missing = set(data) - names, names - set(data)
    if unknown or missing:
        raise ConfigError(f"section '{name}': unknown {sorted(unknown)}, missing {sorted(missing)}")
    hints = get_type_hints(cls)
    return cls(**{k: _typed(v, hints[k], f"{name}.{k}") for k, v in data.items()})


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
