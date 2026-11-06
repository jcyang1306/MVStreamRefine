"""Typed P4 keyframe-memory configuration (`configs/keyframes.yaml`)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from mvsr.paths import repo_root
from mvsr.yaml_config import ConfigError, load_section

DEFAULT_CONFIG = "configs/keyframes.yaml"


@dataclass(frozen=True)
class SelectionConfig:
    min_frame_spacing: int
    max_frame_gap: int
    min_keypoints: int
    min_geometry_points: int
    min_pose_overlap: float
    min_covisibility: float
    min_viewpoint_translation_m: float
    min_viewpoint_angle_deg: float
    min_appearance_hamming: int
    feature_grid_size: int
    min_spatial_novelty: float


@dataclass(frozen=True)
class RetrievalConfig:
    max_local_keyframes: int
    max_global_keyframes: int
    visibility_voxel_m: float
    max_visibility_voxels: int
    viewpoint_weight: float
    appearance_weight: float
    covisibility_weight: float
    min_viewpoint_separation_deg: float


@dataclass(frozen=True)
class PersistenceConfig:
    verify_raw_refs: bool


@dataclass(frozen=True)
class KeyframeBenchmarkConfig:
    reproducibility_frames: int


@dataclass(frozen=True)
class KeyframeGateConfig:
    max_keyframes: int
    min_best_anchor_hit_rate: float
    min_revisit_reuse_ratio: float


@dataclass(frozen=True)
class KeyframeConfig:
    selection: SelectionConfig
    retrieval: RetrievalConfig
    persistence: PersistenceConfig
    benchmark: KeyframeBenchmarkConfig
    gate: KeyframeGateConfig
    source_path: str
    source_sha256: str

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def load_keyframe_config(path: str | Path | None = None) -> KeyframeConfig:
    p = Path(path) if path is not None else repo_root() / DEFAULT_CONFIG
    text = p.read_bytes()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: not a mapping")
    cfg = KeyframeConfig(
        selection=load_section(raw, SelectionConfig, "selection"),
        retrieval=load_section(raw, RetrievalConfig, "retrieval"),
        persistence=load_section(raw, PersistenceConfig, "persistence"),
        benchmark=load_section(raw, KeyframeBenchmarkConfig, "benchmark"),
        gate=load_section(raw, KeyframeGateConfig, "gate"),
        source_path=str(p),
        source_sha256=hashlib.sha256(text).hexdigest(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: KeyframeConfig) -> None:
    s = cfg.selection
    if (
        min(
            s.min_frame_spacing,
            s.max_frame_gap,
            s.min_keypoints,
            s.min_geometry_points,
            s.min_appearance_hamming,
            s.feature_grid_size,
        )
        < 1
    ):
        raise ConfigError("selection integer thresholds must be >= 1")
    if (
        not 0 <= s.min_pose_overlap <= 1
        or not 0 <= s.min_covisibility <= 1
        or min(s.min_viewpoint_translation_m, s.min_viewpoint_angle_deg) <= 0
        or s.min_appearance_hamming > 256
        or not 0 <= s.min_spatial_novelty <= 1
    ):
        raise ConfigError("invalid selection thresholds")
    r = cfg.retrieval
    if (
        min(r.max_local_keyframes, r.max_global_keyframes, r.max_visibility_voxels) < 1
        or r.max_local_keyframes > r.max_global_keyframes
        or r.visibility_voxel_m <= 0
        or min(
            r.viewpoint_weight,
            r.appearance_weight,
            r.covisibility_weight,
            r.min_viewpoint_separation_deg,
        )
        < 0
    ):
        raise ConfigError("invalid retrieval settings")
    if cfg.benchmark.reproducibility_frames < 1:
        raise ConfigError("benchmark.reproducibility_frames must be >= 1")
    g = cfg.gate
    if (
        g.max_keyframes < 1
        or not 0 <= g.min_best_anchor_hit_rate <= 1
        or not 0 <= g.min_revisit_reuse_ratio <= 1
    ):
        raise ConfigError("invalid keyframe gate settings")
