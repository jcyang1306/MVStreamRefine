"""P5 object-centric mapping benchmark and I3 gate."""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mvsr import paths
from mvsr.core import codec
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.se3 import PoseSE3
from mvsr.core.types import Keyframe, make_session_id
from mvsr.io.atomic import atomic_write_json, sha256_file
from mvsr.io.session_writer import MANIFEST_NAME
from mvsr.keyframes.config import KeyframeConfig
from mvsr.keyframes.manager import KeyframeManager
from mvsr.mapping.config import MappingConfig
from mvsr.mapping.map import ObjectMap, pose_from_keyframe
from mvsr.mapping.open3d_backend import (
    GeometryResult,
    create_volume,
    extract,
    integrate,
    pointcloud_from_geometry,
)
from mvsr.observation.masks import MaskCache
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.stats import summary
from mvsr.software import software_info
from mvsr.tracking.benchmark import STATIC_GT, _spread

R_BOTH = "R-BOTH"


@dataclass(frozen=True)
class MappingBenchmarkResult:
    report: dict[str, Any]
    object_map: ObjectMap
    geometry: GeometryResult
    train_keyframes: tuple[Keyframe, ...]


def find_latest_keyframe_memory(root: Path, session_id: str) -> Path:
    candidates = sorted(
        (
            path
            for path in (root / "runs").glob(f"*/keyframes/{session_id}")
            if (path / MANIFEST_NAME).is_file()
        ),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no P4 keyframe memory for {session_id}; run `mvsr p4-gate`")
    return candidates[0]


def _sample_keyframes(keyframes: tuple[Keyframe, ...], maximum: int) -> tuple[Keyframe, ...]:
    if len(keyframes) <= maximum:
        return keyframes
    indices = np.linspace(0, len(keyframes) - 1, maximum, dtype=np.intp)
    return tuple(keyframes[int(index)] for index in indices)


def _build_geometry(
    manager: KeyframeManager,
    keyframes: tuple[Keyframe, ...],
    cfg: MappingConfig,
    bundle: Any,
    pose_fn: Callable[[Keyframe], PoseSE3],
) -> GeometryResult:
    volume = create_volume(cfg.tsdf)
    for keyframe in keyframes:
        integrate(
            volume,
            manager.observation(keyframe.keyframe_id),
            manager.reader.read_frame("wrist", keyframe.header.frame_id),
            bundle,
            pose_fn(keyframe),
            cfg.tsdf,
        )
    return extract(volume, color=cfg.tsdf.color)


def _query_points(
    manager: KeyframeManager,
    keyframes: tuple[Keyframe, ...],
    maximum: int,
) -> np.ndarray:
    points: list[np.ndarray] = []
    each = max(1, maximum // max(1, len(keyframes)))
    for keyframe in keyframes:
        observation = manager.observation(keyframe.keyframe_id)
        values = keyframe.T_cam_obj_current.inverse().transform_points(observation.points_cam_m)
        if len(values) > each:
            indices = np.linspace(0, len(values) - 1, each, dtype=np.intp)
            values = values[indices]
        points.append(values)
    return np.concatenate(points) if points else np.zeros((0, 3))


def _residual(
    geometry: GeometryResult,
    query_points: np.ndarray,
    threshold_m: float,
) -> dict[str, Any] | None:
    if not len(geometry.points_obj_m) or not len(query_points):
        return None
    import open3d as o3d

    query = o3d.geometry.PointCloud()
    query.points = o3d.utility.Vector3dVector(query_points)
    distances = np.asarray(
        query.compute_point_cloud_distance(pointcloud_from_geometry(geometry)),
        dtype=np.float64,
    )
    stats = summary(distances, 1e3)
    return {
        "distance_mm": stats,
        "coverage": float((distances <= threshold_m).mean()),
    }


def _geometry_digest(geometry: GeometryResult) -> str:
    digest = hashlib.sha256()
    for array in (
        geometry.points_obj_m,
        geometry.normals_obj,
        geometry.vertices_obj_m,
        geometry.triangles,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    if geometry.colors_u8 is not None:
        digest.update(np.ascontiguousarray(geometry.colors_u8).tobytes())
    return digest.hexdigest()


def run_benchmark(
    session_dir: str | Path,
    memory_dir: str | Path,
    mapping_cfg: MappingConfig,
    keyframe_cfg: KeyframeConfig,
) -> MappingBenchmarkResult:
    source = ReplaySource(session_dir)
    memory_manifest = json.loads((Path(memory_dir) / MANIFEST_NAME).read_text())
    revision = str(memory_manifest["mask_revision"])
    masks = MaskCache(session_dir, revision)
    manager = KeyframeManager.load(memory_dir, session_dir, masks, keyframe_cfg)
    gauge = manager.T_base_obj_init
    if gauge is None:
        raise ValueError("P4 memory has no object gauge")
    selected = _sample_keyframes(manager.keyframes, mapping_cfg.benchmark.max_keyframes)
    heldout = tuple(
        keyframe
        for index, keyframe in enumerate(selected)
        if index > 0 and (index + 1) % mapping_cfg.benchmark.heldout_stride == 0
    )
    heldout_ids = {keyframe.keyframe_id for keyframe in heldout}
    train = tuple(keyframe for keyframe in selected if keyframe.keyframe_id not in heldout_ids)
    object_map = ObjectMap(
        session_dir,
        source.bundle,
        manager,
        gauge,
        mapping_cfg,
    )
    geometry = object_map.build_provisional(list(train))
    repeat = object_map.build_provisional(list(train))
    query = _query_points(manager, heldout, mapping_cfg.benchmark.surface_sample_points)
    heldout_result = _residual(geometry, query, mapping_cfg.gate.max_heldout_residual_p95_mm / 1e3)
    wrong = _build_geometry(
        manager,
        train,
        mapping_cfg,
        source.bundle,
        lambda keyframe: PoseSE3(keyframe.T_cam_obj_current.inverse().matrix, "cam", "obj"),
    )
    wrong_result = _residual(wrong, query, mapping_cfg.gate.max_heldout_residual_p95_mm / 1e3)
    scenario = source.reader.manifest.get("scenario") or {}
    robot_result = None
    if scenario.get("gt_type") == STATIC_GT:
        robot = _build_geometry(
            manager,
            train,
            mapping_cfg,
            source.bundle,
            lambda keyframe: (
                keyframe.T_base_cam_prior.inverse() @ gauge
                if keyframe.T_base_cam_prior is not None
                else keyframe.T_cam_obj_current
            ),
        )
        robot_result = _residual(robot, query, mapping_cfg.gate.max_heldout_residual_p95_mm / 1e3)
    base_poses = [
        keyframe.T_base_cam_prior @ keyframe.T_cam_obj_current
        for keyframe in selected
        if keyframe.T_base_cam_prior is not None
    ]
    tracking_spread = _spread(base_poses, gauge)
    translation_p95 = None if tracking_spread is None else tracking_spread["translation_mm"]["p95"]
    rotation_p95 = None if tracking_spread is None else tracking_spread["rotation_deg"]["p95"]
    tracking_authorized = (
        translation_p95 is not None
        and rotation_p95 is not None
        and translation_p95 <= mapping_cfg.gate.max_tracking_translation_p95_mm
        and rotation_p95 <= mapping_cfg.gate.max_tracking_rotation_p95_deg
    )
    current_p95 = None if heldout_result is None else heldout_result["distance_mm"]["p95"]
    wrong_p95 = None if wrong_result is None else wrong_result["distance_mm"]["p95"]
    wrong_ratio = (
        None
        if current_p95 in (None, 0)
        else (wrong_p95 if wrong_p95 is not None else mapping_cfg.tsdf.depth_trunc_m * 1e3)
        / current_p95
    )
    report = {
        "session_id": source.session_id,
        "host": source.reader.manifest.get("host"),
        "manifest_sha256": sha256_file(Path(source.reader.dir) / MANIFEST_NAME),
        "scenario": scenario,
        "mask_revision": revision,
        "keyframe_memory": {
            "path": str(memory_dir),
            "manifest_sha256": sha256_file(Path(memory_dir) / MANIFEST_NAME),
            "digest": manager.digest(),
        },
        "keyframes": {
            "available": len(manager.keyframes),
            "train": [keyframe.keyframe_id for keyframe in train],
            "heldout": [keyframe.keyframe_id for keyframe in heldout],
        },
        "tracking_spread": tracking_spread,
        "tracking_authorized_for_commit": tracking_authorized,
        "geometry": {
            "points": len(geometry.points_obj_m),
            "vertices": len(geometry.vertices_obj_m),
            "triangles": len(geometry.triangles),
            "bounds_min_m": (
                geometry.points_obj_m.min(axis=0).tolist() if len(geometry.points_obj_m) else None
            ),
            "bounds_max_m": (
                geometry.points_obj_m.max(axis=0).tolist() if len(geometry.points_obj_m) else None
            ),
            "digest": _geometry_digest(geometry),
            "repeat_digest": _geometry_digest(repeat),
            "deterministic": _geometry_digest(geometry) == _geometry_digest(repeat),
        },
        "heldout": heldout_result,
        "robot_prior_static_baseline": robot_result,
        "wrong_inverse_baseline": wrong_result,
        "wrong_inverse_error_ratio": wrong_ratio,
        "commit_mode": "PROVISIONAL_DIAGNOSTIC",
        "mapping_config_sha256": mapping_cfg.source_sha256,
        "keyframe_config_sha256": keyframe_cfg.source_sha256,
    }
    return MappingBenchmarkResult(report, object_map, geometry, train)


def _check(name: str, ok: bool, value: Any, target: str, hard: bool) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "value": value, "target": target, "hard": hard}


def i3_decision(reports: list[dict[str, Any]], cfg: MappingConfig) -> dict[str, Any]:
    tags = {tag for report in reports for tag in ((report.get("scenario") or {}).get("tags") or [])}
    static = [
        report for report in reports if (report.get("scenario") or {}).get("gt_type") == STATIC_GT
    ]
    checks: list[dict[str, Any]] = [
        _check("sessions", bool(reports), len(reports), ">= 1", True),
        _check("r_static_session", bool(static), len(static), ">= 1", True),
        _check("r_both_session", R_BOTH in tags, R_BOTH in tags, "present", True),
    ]
    for report in reports:
        sid = report["session_id"]
        checks.append(
            _check(
                f"{sid}/tracking_authorized",
                report["tracking_authorized_for_commit"],
                report["tracking_authorized_for_commit"],
                (
                    f"p95 <= {cfg.gate.max_tracking_translation_p95_mm} mm / "
                    f"{cfg.gate.max_tracking_rotation_p95_deg} deg"
                ),
                True,
            )
        )
        fused = len(report["keyframes"]["train"])
        checks.append(
            _check(
                f"{sid}/fused_keyframes",
                fused >= cfg.gate.min_fused_keyframes,
                fused,
                f">= {cfg.gate.min_fused_keyframes}",
                True,
            )
        )
        points = report["geometry"]["points"]
        checks.append(
            _check(
                f"{sid}/geometry_points",
                points >= cfg.gate.min_geometry_points,
                points,
                f">= {cfg.gate.min_geometry_points}",
                True,
            )
        )
        checks.append(
            _check(
                f"{sid}/deterministic",
                report["geometry"]["deterministic"],
                report["geometry"]["deterministic"],
                "identical TSDF extraction digest",
                True,
            )
        )
        heldout = report.get("heldout")
        p95 = None if heldout is None else heldout["distance_mm"]["p95"]
        coverage = None if heldout is None else heldout["coverage"]
        checks.append(
            _check(
                f"{sid}/heldout_p95_mm",
                p95 is not None and p95 <= cfg.gate.max_heldout_residual_p95_mm,
                p95,
                f"<= {cfg.gate.max_heldout_residual_p95_mm}",
                False,
            )
        )
        checks.append(
            _check(
                f"{sid}/heldout_coverage",
                coverage is not None and coverage >= cfg.gate.min_heldout_coverage,
                coverage,
                f">= {cfg.gate.min_heldout_coverage}",
                False,
            )
        )
        ratio = report.get("wrong_inverse_error_ratio")
        checks.append(
            _check(
                f"{sid}/wrong_inverse_error_ratio",
                ratio is not None and ratio >= cfg.gate.min_wrong_extrinsic_error_ratio,
                ratio,
                f">= {cfg.gate.min_wrong_extrinsic_error_ratio}",
                True,
            )
        )
    hard = [check["name"] for check in checks if check["hard"] and not check["ok"]]
    soft = [check["name"] for check in checks if not check["hard"] and not check["ok"]]
    return {
        "gate": "I3",
        "decision": "BLOCKED" if hard else ("PASS-WITH-LIMITS" if soft else "PASS"),
        "checks": checks,
        "failed_hard": hard,
        "missed_targets": soft,
        "scope_limits": [
            "geometry is the reachable-viewpoint partial surface, not a closed object",
            "provisional artifacts are never exposed through committed MapView",
            "R-BOTH and independently accurate poses are required before committed fusion",
        ],
        "targets_are_provisional": True,
    }


def write_run(
    results: list[MappingBenchmarkResult],
    decision: dict[str, Any],
    cfg: MappingConfig,
    root: Path | None = None,
) -> Path:
    reports = [result.report for result in results]
    hosts = sorted({str(report["host"]) for report in reports if report.get("host")})
    run_id = make_session_id(hosts[0] if len(hosts) == 1 else socket.gethostname())
    directory = (root or paths.data_root()) / "runs" / run_id
    directory.mkdir(parents=True)
    final_decision = i3_decision(reports, cfg)
    decision.clear()
    decision.update(final_decision)
    for result in results:
        session_dir = directory / "mapping" / result.report["session_id"]
        provisional_files = result.object_map.write_provisional(session_dir, result.geometry)
        result.report["artifacts"] = {
            "mode": "PROVISIONAL_DIAGNOSTIC",
            "files": provisional_files,
        }
        if final_decision["decision"] == "PASS":
            for keyframe in result.train_keyframes:
                result.object_map.commit(keyframe, pose_from_keyframe(keyframe))
            snapshot = result.object_map.publish(session_dir / "committed", run_id)
            result.report["commit_mode"] = "COMMITTED"
            result.report["snapshot"] = codec.to_json_dict(snapshot)
    body = {
        "schema_version": SCHEMA_VERSION,
        "type": "MappingReport",
        "run_id": run_id,
        "sessions": reports,
        "i3": final_decision,
    }
    atomic_write_json(directory / "mapping_report.json", body)
    files = {
        str(path.relative_to(directory)): sha256_file(path)
        for path in directory.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "P5Run",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "processing_host": socket.gethostname(),
        "software": software_info(),
        "inputs": [
            {
                "session_id": report["session_id"],
                "manifest_sha256": report["manifest_sha256"],
                "mask_revision": report["mask_revision"],
                "keyframe_memory_manifest_sha256": report["keyframe_memory"]["manifest_sha256"],
            }
            for report in reports
        ],
        "config": {
            "path": cfg.source_path,
            "sha256": cfg.source_sha256,
            "values": cfg.snapshot(),
        },
        "files": files,
    }
    atomic_write_json(directory / MANIFEST_NAME, manifest)
    return directory
