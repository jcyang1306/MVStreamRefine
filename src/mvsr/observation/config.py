"""P2 configuration (`configs/observation.yaml`), validated into frozen dataclasses."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from mvsr.paths import repo_root
from mvsr.yaml_config import ConfigError, load_section

DEFAULT_CONFIG = "configs/observation.yaml"


@dataclass(frozen=True)
class Sam2Config:
    checkpoint: str
    checkpoint_sha256: str
    model_type: str
    device: str
    memory_frames: int | None
    min_mask_area_px: int
    low_confidence_score: float

    def checkpoint_path(self) -> Path:
        p = Path(self.checkpoint)
        return p if p.is_absolute() else repo_root() / p


@dataclass(frozen=True)
class DepthConfig:
    min_depth_m: float
    max_depth_m: float
    mask_erode_px: int
    edge_jump_rel: float
    noise_k: float
    sigma_a_m: float
    sigma_b_per_m: float
    point_stride: int


@dataclass(frozen=True)
class FeatureConfig:
    type: str
    n_features: int
    fast_threshold: int
    scale_factor: float
    n_levels: int
    edge_threshold: int
    patch_size: int
    safe_erode_px: int


@dataclass(frozen=True)
class TrackConfig:
    klt_window_px: int
    klt_levels: int
    fb_max_px: float
    max_occluded_frames: int
    target_tracks: int
    min_spacing_px: int
    max_descriptor_drift_bits: int | None


@dataclass(frozen=True)
class CorrespondenceConfig:
    ratio: float
    max_hamming: int
    sigma_px: float
    ransac_iterations: int
    ransac_rigid_m: float
    ransac_epipolar_px: float
    min_model_matches: int
    seed: int


@dataclass(frozen=True)
class BenchmarkConfig:
    frame_gaps: list[int]
    pair_stride: int
    true_match_m: float
    true_match_px: float
    mask_sample_points: int
    reproducibility_frames: int


@dataclass(frozen=True)
class ObservationGateConfig:
    min_mask_coverage: float
    min_mask_agreement: float
    min_valid_depth_ratio: float
    min_match_precision: float
    min_tracked_keypoints: int


@dataclass(frozen=True)
class ObservationConfig:
    sam2: Sam2Config
    depth: DepthConfig
    features: FeatureConfig
    tracks: TrackConfig
    correspondence: CorrespondenceConfig
    benchmark: BenchmarkConfig
    gate: ObservationGateConfig
    source_path: str
    source_sha256: str

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def load_observation_config(path: str | Path | None = None) -> ObservationConfig:
    p = Path(path) if path is not None else repo_root() / DEFAULT_CONFIG
    text = p.read_bytes()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: not a mapping")
    cfg = ObservationConfig(
        sam2=load_section(raw, Sam2Config, "sam2"),
        depth=load_section(raw, DepthConfig, "depth"),
        features=load_section(raw, FeatureConfig, "features"),
        tracks=load_section(raw, TrackConfig, "tracks"),
        correspondence=load_section(raw, CorrespondenceConfig, "correspondence"),
        benchmark=load_section(raw, BenchmarkConfig, "benchmark"),
        gate=load_section(raw, ObservationGateConfig, "gate"),
        source_path=str(p),
        source_sha256=hashlib.sha256(text).hexdigest(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: ObservationConfig) -> None:
    s = cfg.sam2
    if s.model_type not in ("tiny", "small", "base_plus", "large"):
        raise ConfigError("sam2.model_type must be tiny|small|base_plus|large")
    if len(s.checkpoint_sha256) != 64:
        raise ConfigError("sam2.checkpoint_sha256 must be a sha256 hex digest")
    if s.min_mask_area_px < 1 or not 0.0 <= s.low_confidence_score <= 1.0:
        raise ConfigError("sam2: min_mask_area_px >= 1, low_confidence_score in [0, 1]")
    d = cfg.depth
    if not 0 < d.min_depth_m < d.max_depth_m:
        raise ConfigError("depth: 0 < min_depth_m < max_depth_m")
    if d.mask_erode_px < 0 or d.edge_jump_rel <= 0 or d.noise_k <= 0 or d.point_stride < 1:
        raise ConfigError("depth: mask_erode_px >= 0, edge_jump_rel/noise_k > 0, point_stride >= 1")
    if d.sigma_a_m < 0 or d.sigma_b_per_m < 0 or d.sigma_a_m + d.sigma_b_per_m <= 0:
        raise ConfigError("depth: sigma_a_m, sigma_b_per_m >= 0 and not both 0")
    f = cfg.features
    if f.type != "orb":
        raise ConfigError(f"features.type {f.type!r}: only 'orb' is implemented")
    if f.n_features < 1 or f.patch_size < 7 or f.edge_threshold < f.patch_size // 2 + 3:
        raise ConfigError("features: n_features >= 1, patch_size >= 7, edge_threshold >= patch/2+3")
    t = cfg.tracks
    if t.klt_window_px < 5 or t.klt_levels < 0 or t.fb_max_px <= 0 or t.target_tracks < 1:
        raise ConfigError("tracks: klt_window_px >= 5, klt_levels >= 0, fb_max_px > 0, target >= 1")
    c = cfg.correspondence
    if not 0 < c.ratio <= 1 or c.ransac_iterations < 1 or c.min_model_matches < 3:
        raise ConfigError("correspondence: ratio in (0, 1], ransac_iterations >= 1, min_model >= 3")
    if c.ransac_rigid_m <= 0 or c.ransac_epipolar_px <= 0 or c.sigma_px <= 0:
        raise ConfigError("correspondence: thresholds and sigma_px must be > 0")
    b = cfg.benchmark
    if not b.frame_gaps or min(b.frame_gaps) < 1:
        raise ConfigError("benchmark.frame_gaps must be a non-empty list of ints >= 1")
    if b.pair_stride < 1 or b.true_match_m <= 0 or b.true_match_px <= 0:
        raise ConfigError("benchmark: pair_stride >= 1, true_match_m/px > 0")
    g = cfg.gate
    if not all(
        0.0 <= v <= 1.0
        for v in (
            g.min_mask_coverage,
            g.min_mask_agreement,
            g.min_valid_depth_ratio,
            g.min_match_precision,
        )
    ):
        raise ConfigError("gate ratios must be in [0, 1]")
