"""Typed P5 mapping configuration (`configs/mapping.yaml`)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from mvsr.paths import repo_root
from mvsr.yaml_config import ConfigError, load_section

DEFAULT_CONFIG = "configs/mapping.yaml"


@dataclass(frozen=True)
class TsdfConfig:
    voxel_length_m: float
    sdf_trunc_m: float
    depth_trunc_m: float
    color: bool


@dataclass(frozen=True)
class CommitConfig:
    min_geometry_points: int
    min_overlap: float
    max_rmse_m: float
    require_pose_version_match: bool


@dataclass(frozen=True)
class OutputConfig:
    min_point_count: int
    write_mesh: bool
    write_pointcloud: bool


@dataclass(frozen=True)
class MappingBenchmarkConfig:
    max_keyframes: int
    heldout_stride: int
    surface_sample_points: int


@dataclass(frozen=True)
class MappingGateConfig:
    max_tracking_translation_p95_mm: float
    max_tracking_rotation_p95_deg: float
    min_fused_keyframes: int
    min_geometry_points: int
    max_heldout_residual_p95_mm: float
    min_heldout_coverage: float
    min_wrong_extrinsic_error_ratio: float


@dataclass(frozen=True)
class MappingConfig:
    tsdf: TsdfConfig
    commit: CommitConfig
    output: OutputConfig
    benchmark: MappingBenchmarkConfig
    gate: MappingGateConfig
    source_path: str
    source_sha256: str

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def load_mapping_config(path: str | Path | None = None) -> MappingConfig:
    p = Path(path) if path is not None else repo_root() / DEFAULT_CONFIG
    text = p.read_bytes()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: not a mapping")
    cfg = MappingConfig(
        tsdf=load_section(raw, TsdfConfig, "tsdf"),
        commit=load_section(raw, CommitConfig, "commit"),
        output=load_section(raw, OutputConfig, "output"),
        benchmark=load_section(raw, MappingBenchmarkConfig, "benchmark"),
        gate=load_section(raw, MappingGateConfig, "gate"),
        source_path=str(p),
        source_sha256=hashlib.sha256(text).hexdigest(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: MappingConfig) -> None:
    t = cfg.tsdf
    if (
        min(t.voxel_length_m, t.sdf_trunc_m, t.depth_trunc_m) <= 0
        or t.sdf_trunc_m < 2 * t.voxel_length_m
    ):
        raise ConfigError("tsdf lengths must be positive and truncation >= 2 voxels")
    c = cfg.commit
    if c.min_geometry_points < 1 or not 0 <= c.min_overlap <= 1 or c.max_rmse_m <= 0:
        raise ConfigError("invalid commit thresholds")
    o = cfg.output
    if o.min_point_count < 1 or not (o.write_mesh or o.write_pointcloud):
        raise ConfigError("output requires a geometry type and positive point count")
    b = cfg.benchmark
    if min(b.max_keyframes, b.heldout_stride, b.surface_sample_points) < 1:
        raise ConfigError("benchmark counts must be >= 1")
    g = cfg.gate
    if (
        min(
            g.max_tracking_translation_p95_mm,
            g.max_tracking_rotation_p95_deg,
            g.min_fused_keyframes,
            g.min_geometry_points,
            g.max_heldout_residual_p95_mm,
            g.min_wrong_extrinsic_error_ratio,
        )
        <= 0
    ):
        raise ConfigError("mapping gate thresholds must be > 0")
    if not 0 <= g.min_heldout_coverage <= 1:
        raise ConfigError("gate.min_heldout_coverage must be in [0, 1]")
