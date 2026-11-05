"""Typed P3 configuration (`configs/tracking.yaml`)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from mvsr.paths import repo_root
from mvsr.yaml_config import ConfigError, load_section

DEFAULT_CONFIG = "configs/tracking.yaml"


@dataclass(frozen=True)
class InitializationConfig:
    min_geometry_points: int
    min_keypoints_with_depth: int


@dataclass(frozen=True)
class HypothesisConfig:
    min_inliers: int
    max_local_keyframes: int
    max_icp_hypotheses: int
    use_recent_tracks: bool


@dataclass(frozen=True)
class IcpConfig:
    enabled: bool
    max_points: int
    iterations: list[int]
    max_correspondence_m: list[float]
    huber_delta_m: float
    min_correspondences: int
    normal_angle_deg: float
    max_refinement_m: float
    max_refinement_deg: float


@dataclass(frozen=True)
class ValidityConfig:
    min_overlap: float
    max_rmse_m: float
    min_hessian_eig: float
    max_correction_m: float
    max_correction_deg: float
    prediction_only_frames: int
    lost_after_frames: int


@dataclass(frozen=True)
class KeyframeConfig:
    max_keyframes: int
    max_frame_gap: int
    min_anchor_overlap: float
    max_anchor_translation_m: float
    max_anchor_rotation_deg: float


@dataclass(frozen=True)
class TrackingBenchmarkConfig:
    reproducibility_frames: int


@dataclass(frozen=True)
class TrackingGateConfig:
    min_valid_ratio: float
    max_static_translation_p95_mm: float
    max_static_rotation_p95_deg: float
    max_failure_ratio: float


@dataclass(frozen=True)
class TrackingConfig:
    initialization: InitializationConfig
    hypothesis: HypothesisConfig
    icp: IcpConfig
    validity: ValidityConfig
    keyframes: KeyframeConfig
    benchmark: TrackingBenchmarkConfig
    gate: TrackingGateConfig
    source_path: str
    source_sha256: str

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def load_tracking_config(path: str | Path | None = None) -> TrackingConfig:
    p = Path(path) if path is not None else repo_root() / DEFAULT_CONFIG
    text = p.read_bytes()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: not a mapping")
    cfg = TrackingConfig(
        initialization=load_section(raw, InitializationConfig, "initialization"),
        hypothesis=load_section(raw, HypothesisConfig, "hypothesis"),
        icp=load_section(raw, IcpConfig, "icp"),
        validity=load_section(raw, ValidityConfig, "validity"),
        keyframes=load_section(raw, KeyframeConfig, "keyframes"),
        benchmark=load_section(raw, TrackingBenchmarkConfig, "benchmark"),
        gate=load_section(raw, TrackingGateConfig, "gate"),
        source_path=str(p),
        source_sha256=hashlib.sha256(text).hexdigest(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: TrackingConfig) -> None:
    init = cfg.initialization
    if min(init.min_geometry_points, init.min_keypoints_with_depth) < 1:
        raise ConfigError("initialization counts must be >= 1")
    hyp = cfg.hypothesis
    if min(hyp.min_inliers, hyp.max_local_keyframes, hyp.max_icp_hypotheses) < 1:
        raise ConfigError("hypothesis counts must be >= 1")
    icp = cfg.icp
    if (
        icp.max_points < 3
        or not icp.iterations
        or len(icp.iterations) != len(icp.max_correspondence_m)
        or min(icp.iterations) < 1
        or min(icp.max_correspondence_m) <= 0
        or icp.huber_delta_m <= 0
        or icp.min_correspondences < 6
        or not 0 < icp.normal_angle_deg <= 90
        or min(icp.max_refinement_m, icp.max_refinement_deg) <= 0
    ):
        raise ConfigError("invalid icp settings")
    v = cfg.validity
    if (
        not 0 <= v.min_overlap <= 1
        or min(v.max_rmse_m, v.min_hessian_eig, v.max_correction_m, v.max_correction_deg) <= 0
        or min(v.prediction_only_frames, v.lost_after_frames) < 1
        or v.prediction_only_frames > v.lost_after_frames
    ):
        raise ConfigError("invalid validity settings")
    k = cfg.keyframes
    if (
        min(k.max_keyframes, k.max_frame_gap) < 1
        or not 0 <= k.min_anchor_overlap <= 1
        or min(k.max_anchor_translation_m, k.max_anchor_rotation_deg) <= 0
    ):
        raise ConfigError("invalid keyframe settings")
    if cfg.benchmark.reproducibility_frames < 1:
        raise ConfigError("benchmark.reproducibility_frames must be >= 1")
    g = cfg.gate
    if (
        not 0 <= g.min_valid_ratio <= 1
        or not 0 <= g.max_failure_ratio <= 1
        or min(g.max_static_translation_p95_mm, g.max_static_rotation_p95_deg) <= 0
    ):
        raise ConfigError("invalid gate settings")
