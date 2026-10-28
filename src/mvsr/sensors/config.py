"""P1 configuration (`configs/sensors.yaml`), validated into frozen dataclasses."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from mvsr.paths import repo_root
from mvsr.yaml_config import ConfigError, load_section

DEFAULT_CONFIG = "configs/sensors.yaml"


@dataclass(frozen=True)
class ClockConfig:
    device_counter_period_us: int
    max_rate_error_ppm: float
    min_envelope_points: int
    fallback_latency_ns: int


@dataclass(frozen=True)
class SyncConfig:
    max_interp_gap_ns: int


@dataclass(frozen=True)
class RgbdConfig:
    max_color_depth_dt_ms: dict[str, float | None]
    skip_mismatched: bool

    def limit_ms(self, camera_id: str) -> float | None:
        return self.max_color_depth_dt_ms.get(camera_id)


@dataclass(frozen=True)
class LiveConfig:
    robot_buffer_s: float
    robot_wait_ms: int
    first_frame_timeout_ms: int
    frame_timeout_ms: int
    max_consecutive_timeouts: int
    warmup_frames: int


@dataclass(frozen=True)
class BoardConfig:
    inner_cols: int
    inner_rows: int
    square_mm: float | None
    max_reproj_rms_px: float
    stride: int
    workers: int


@dataclass(frozen=True)
class DepthCheckConfig:
    edge_margin_px: int
    corner_window_px: int
    bin_m: float
    min_valid_ratio: float


@dataclass(frozen=True)
class GateConfig:
    min_board_frames: int
    min_interpolated_ratio: float
    min_rgbd_valid_ratio: float
    max_clock_residual_p95_ms: float
    max_board_spread_p95_mm: float
    max_board_spread_p95_deg: float
    max_handeye_reproj_p95_px: float


@dataclass(frozen=True)
class SensorsConfig:
    clock: ClockConfig
    sync: SyncConfig
    rgbd: RgbdConfig
    live: LiveConfig
    board: BoardConfig
    depth: DepthCheckConfig
    gate: GateConfig
    source_path: str
    source_sha256: str

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def load_sensors_config(path: str | Path | None = None) -> SensorsConfig:
    p = Path(path) if path is not None else repo_root() / DEFAULT_CONFIG
    text = p.read_bytes()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: not a mapping")
    cfg = SensorsConfig(
        clock=load_section(raw, ClockConfig, "clock"),
        sync=load_section(raw, SyncConfig, "sync"),
        rgbd=load_section(raw, RgbdConfig, "rgbd"),
        live=load_section(raw, LiveConfig, "live"),
        board=load_section(raw, BoardConfig, "board"),
        depth=load_section(raw, DepthCheckConfig, "depth"),
        gate=load_section(raw, GateConfig, "gate"),
        source_path=str(p),
        source_sha256=hashlib.sha256(text).hexdigest(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: SensorsConfig) -> None:
    c = cfg.clock
    if c.device_counter_period_us <= 0 or c.max_rate_error_ppm <= 0 or c.min_envelope_points < 2:
        raise ConfigError("clock: period/max_rate_error_ppm must be > 0, min_envelope_points >= 2")
    if c.fallback_latency_ns < 0:
        raise ConfigError("clock.fallback_latency_ns must be >= 0")
    if cfg.sync.max_interp_gap_ns <= 0:
        raise ConfigError("sync.max_interp_gap_ns must be > 0")
    for cid, lim in cfg.rgbd.max_color_depth_dt_ms.items():
        if cid not in ("wrist", "head") or (lim is not None and lim <= 0):
            raise ConfigError(
                f"rgbd.max_color_depth_dt_ms.{cid} must be a camera id with a limit > 0 or null"
            )
    lv = cfg.live
    if lv.robot_buffer_s <= 0 or lv.robot_wait_ms < 0 or lv.max_consecutive_timeouts < 1:
        raise ConfigError(
            "live: robot_buffer_s > 0, robot_wait_ms >= 0, max_consecutive_timeouts >= 1"
        )
    if min(lv.first_frame_timeout_ms, lv.frame_timeout_ms) <= 0 or lv.warmup_frames < 0:
        raise ConfigError("live: frame timeouts must be > 0 and warmup_frames >= 0")
    b = cfg.board
    if min(b.inner_cols, b.inner_rows) < 2 or (b.inner_cols + b.inner_rows) % 2 == 0:
        raise ConfigError(
            "board: inner_cols/rows >= 2 and cols + rows odd (else 180 deg ambiguous)"
        )
    if b.square_mm is not None and b.square_mm <= 0:
        raise ConfigError("board.square_mm must be > 0 or null")
    if b.max_reproj_rms_px <= 0 or b.stride < 1 or b.workers < 1:
        raise ConfigError("board: max_reproj_rms_px > 0, stride >= 1, workers >= 1")
    d = cfg.depth
    if d.edge_margin_px < 0 or d.corner_window_px < 0 or d.bin_m <= 0:
        raise ConfigError("depth: edge_margin_px/corner_window_px >= 0, bin_m > 0")
    if not 0.0 <= d.min_valid_ratio <= 1.0:
        raise ConfigError("depth.min_valid_ratio must be in [0, 1]")
    g = cfg.gate
    if g.min_board_frames < 1 or not 0.0 <= g.min_interpolated_ratio <= 1.0:
        raise ConfigError("gate: min_board_frames >= 1, min_interpolated_ratio in [0, 1]")
    if not 0.0 <= g.min_rgbd_valid_ratio <= 1.0:
        raise ConfigError("gate.min_rgbd_valid_ratio must be in [0, 1]")
    if (
        min(
            g.max_clock_residual_p95_ms,
            g.max_board_spread_p95_mm,
            g.max_board_spread_p95_deg,
            g.max_handeye_reproj_p95_px,
        )
        <= 0
    ):
        raise ConfigError("gate: max_* targets must be > 0")
