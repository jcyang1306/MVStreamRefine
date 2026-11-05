"""P3 deterministic replay benchmark and I2 decision."""

from __future__ import annotations

import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mvsr import paths
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.se3 import PoseSE3, so3_log
from mvsr.core.types import PoseEstimate, PoseStatus, make_session_id
from mvsr.io.atomic import atomic_write_json, sha256_file
from mvsr.io.session_writer import MANIFEST_NAME
from mvsr.observation.builder import ObservationBuilder
from mvsr.observation.config import ObservationConfig
from mvsr.observation.correspondence import KeypointView, match_observations
from mvsr.observation.masks import MaskCache
from mvsr.sensors.config import SensorsConfig
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.stats import summary
from mvsr.software import software_info
from mvsr.tracking.config import TrackingConfig
from mvsr.tracking.icp import point_to_plane_icp
from mvsr.tracking.session import TrackingSession, pose_digest
from mvsr.tracking.tracker import ObjectTracker

STATIC_GT = "constant_T_base_obj"
MARKER_TAG = "R-MARKER"
BOTH_TAG = "R-BOTH"


def _errors(poses: list[PoseSE3], reference: PoseSE3) -> tuple[list[float], list[float]]:
    translation, rotation = [], []
    for pose in poses:
        delta = pose.matrix @ np.linalg.inv(reference.matrix)
        translation.append(float(np.linalg.norm(delta[:3, 3])))
        rotation.append(float(np.degrees(np.linalg.norm(so3_log(delta[:3, :3])))))
    return translation, rotation


def _spread(poses: list[PoseSE3], reference: PoseSE3 | None) -> dict[str, Any] | None:
    if not poses or reference is None:
        return None
    translation, rotation = _errors(poses, reference)
    return {
        "translation_mm": summary(translation, 1e3),
        "rotation_deg": summary(rotation),
    }


def _baseline_step(
    previous_obs: Any,
    current_obs: Any,
    previous_pose: PoseSE3,
    source: ReplaySource,
    observation_cfg: ObservationConfig,
) -> PoseSE3 | None:
    for method in ("track", "orb"):
        _, diag = match_observations(
            KeypointView.of(previous_obs),
            KeypointView.of(current_obs),
            source.bundle,
            observation_cfg.correspondence,
            observation_cfg.depth,
            method,
        )
        if diag.rigid_R is not None and diag.rigid_t is not None:
            relative = PoseSE3.from_rt(diag.rigid_R, diag.rigid_t, "cam", "cam")
            return relative @ previous_pose
    return None


def _run_once(
    session_dir: str | Path,
    revision: str,
    tracking_cfg: TrackingConfig,
    observation_cfg: ObservationConfig,
    sensors_cfg: SensorsConfig | None,
    limit: int | None,
    *,
    ablations: bool,
) -> tuple[dict[str, Any], list[PoseEstimate]]:
    source = ReplaySource(session_dir, "wrist", config=sensors_cfg)
    masks = MaskCache(source.reader.dir, revision, "wrist")
    builder = ObservationBuilder(masks, source.bundle, observation_cfg)
    tracker = ObjectTracker(source.bundle, tracking_cfg, observation_cfg)
    pipeline = TrackingSession(tracker)
    ids = [fid for fid in source.frame_ids if fid in masks][:limit]
    poses: list[PoseEstimate] = []
    inserted: dict[int, tuple[int, PoseSE3]] = {}
    anchor_age: list[float] = []
    anchor_translation: list[float] = []
    anchor_rotation: list[float] = []
    chain_cam: PoseSE3 | None = None
    chain_prev_obs: Any = None
    chain_base: list[PoseSE3] = []
    icp_base: list[PoseSE3] = []
    icp_model: tuple[np.ndarray, np.ndarray] | None = None
    icp_last_base: PoseSE3 | None = None
    started = time.monotonic()
    for fid in ids:
        frame = source.read(fid)
        obs = builder.process(frame)
        step = pipeline.process(frame, obs)
        pose = step.pose
        poses.append(pose)
        if step.keyframe_inserted is not None and pose.T_cam_obj is not None:
            inserted[step.keyframe_inserted] = (fid, pose.T_cam_obj)
        anchor_id = int(pose.quality.get("anchor_keyframe_id", -1))
        if pose.T_cam_obj is not None and anchor_id in inserted:
            anchor_frame, anchor_pose = inserted[anchor_id]
            dt, dr = _errors([pose.T_cam_obj], anchor_pose)
            anchor_age.append(float(fid - anchor_frame))
            anchor_translation.append(dt[0])
            anchor_rotation.append(dr[0])
        if not ablations or pose.T_cam_obj is None:
            continue
        if chain_cam is None:
            chain_cam = pose.T_cam_obj
        elif chain_prev_obs is not None:
            chained = _baseline_step(chain_prev_obs, obs, chain_cam, source, observation_cfg)
            if chained is not None:
                chain_cam = chained
        if chain_cam is not None and frame.T_base_cam_prior is not None:
            chain_base.append(frame.T_base_cam_prior @ chain_cam)
        chain_prev_obs = obs
        if icp_model is None and pose.T_cam_obj is not None and obs.normals_cam is not None:
            T_obj_cam = pose.T_cam_obj.inverse()
            icp_model = (
                T_obj_cam.transform_points(obs.points_cam_m),
                obs.normals_cam.astype(np.float64) @ T_obj_cam.R.T,
            )
            icp_last_base = pose.T_base_obj
        elif (
            icp_model is not None
            and icp_last_base is not None
            and frame.T_base_cam_prior is not None
            and obs.normals_cam is not None
        ):
            predicted = frame.T_base_cam_prior.inverse() @ icp_last_base
            result = point_to_plane_icp(
                icp_model[0],
                icp_model[1],
                obs.points_cam_m,
                obs.normals_cam,
                predicted,
                tracking_cfg.icp,
            )
            if result is not None:
                icp_last_base = frame.T_base_cam_prior @ result.T_cam_obj
                icp_base.append(icp_last_base)
    elapsed = time.monotonic() - started
    valid = [p for p in poses if p.status is PoseStatus.VALID]
    usable = [p for p in poses if p.status is not PoseStatus.INVALID]
    base = [p.T_base_obj for p in usable if p.T_base_obj is not None]
    reference = tracker.T_base_obj_init
    scenario = source.reader.manifest.get("scenario") or {}
    static = scenario.get("gt_type") == STATIC_GT
    status = Counter(p.status.value for p in poses)
    states = Counter(p.track_state.value for p in poses)
    reasons = Counter(r.value for p in poses for r in p.reason_codes)
    report = {
        "session_id": source.session_id,
        "host": source.reader.manifest.get("host"),
        "manifest_sha256": sha256_file(Path(source.reader.dir) / MANIFEST_NAME),
        "scenario": scenario,
        "mask_revision": revision,
        "frames": {
            "processed": len(poses),
            "status": dict(sorted(status.items())),
            "track_state": dict(sorted(states.items())),
            "reason_codes": dict(sorted(reasons.items())),
            "valid_ratio": len(valid) / len(poses) if poses else 0.0,
            "usable_ratio": len(usable) / len(poses) if poses else 0.0,
        },
        "keyframes": {
            "inserted": len(inserted),
            "retained": len(pipeline.window.all),
            "anchor_age_frames": summary(anchor_age),
            "distance_to_anchor_translation_mm": summary(anchor_translation, 1e3),
            "distance_to_anchor_rotation_deg": summary(anchor_rotation),
        },
        "static_reference_available": static,
        "static_spread": _spread(base, reference) if static else None,
        "ablations": {
            "frame_to_frame_spread": _spread(chain_base, reference)
            if ablations and static
            else None,
            "icp_only_spread": _spread(icp_base, reference) if ablations and static else None,
            "robot_prior_only": (
                "zero by construction for R-STATIC: fixed T_base_obj_init transformed by each camera prior"
                if ablations and static
                else None
            ),
        },
        "runtime": {
            "wall_s": round(elapsed, 3),
            "fps": round(len(poses) / elapsed, 3) if elapsed else None,
        },
        "pose_digest": pose_digest(poses),
        "tracking_config_sha256": tracking_cfg.source_sha256,
        "observation_config_sha256": observation_cfg.source_sha256,
        "T_base_obj_init": None if reference is None else reference.matrix.tolist(),
    }
    return report, poses


def run_benchmark(
    session_dir: str | Path,
    revision: str,
    tracking_cfg: TrackingConfig,
    observation_cfg: ObservationConfig,
    sensors_cfg: SensorsConfig | None = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    report, _ = _run_once(
        session_dir,
        revision,
        tracking_cfg,
        observation_cfg,
        sensors_cfg,
        limit,
        ablations=True,
    )
    n = min(
        report["frames"]["processed"],
        tracking_cfg.benchmark.reproducibility_frames,
    )
    second, _ = _run_once(
        session_dir,
        revision,
        tracking_cfg,
        observation_cfg,
        sensors_cfg,
        n,
        ablations=False,
    )
    first, _ = _run_once(
        session_dir,
        revision,
        tracking_cfg,
        observation_cfg,
        sensors_cfg,
        n,
        ablations=False,
    )
    report["reproducibility"] = {
        "frames": n,
        "identical": first["pose_digest"] == second["pose_digest"],
        "digest": first["pose_digest"],
    }
    return report


def _check(name: str, ok: bool, value: Any, target: str, hard: bool) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "value": value, "target": target, "hard": hard}


def i2_decision(reports: list[dict[str, Any]], cfg: TrackingConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    static = [r for r in reports if (r.get("scenario") or {}).get("gt_type") == STATIC_GT]
    tags = {tag for r in reports for tag in ((r.get("scenario") or {}).get("tags") or [])}
    checks.append(_check("r_static_session", bool(static), len(static), ">= 1", True))
    for report in reports:
        sid = report["session_id"]
        ratio = report["frames"]["valid_ratio"]
        checks.append(
            _check(
                f"{sid}/valid_ratio",
                ratio >= cfg.gate.min_valid_ratio,
                ratio,
                f">= {cfg.gate.min_valid_ratio}",
                True,
            )
        )
        reproducible = report["reproducibility"]["identical"]
        checks.append(
            _check(
                f"{sid}/reproducible",
                reproducible,
                report["reproducibility"]["frames"],
                "identical pose digests",
                True,
            )
        )
        spread = report.get("static_spread")
        if spread:
            tp95 = spread["translation_mm"]["p95"]
            rp95 = spread["rotation_deg"]["p95"]
            checks.append(
                _check(
                    f"{sid}/static_translation_p95_mm",
                    tp95 <= cfg.gate.max_static_translation_p95_mm,
                    tp95,
                    f"<= {cfg.gate.max_static_translation_p95_mm}",
                    False,
                )
            )
            checks.append(
                _check(
                    f"{sid}/static_rotation_p95_deg",
                    rp95 <= cfg.gate.max_static_rotation_p95_deg,
                    rp95,
                    f"<= {cfg.gate.max_static_rotation_p95_deg}",
                    False,
                )
            )
    checks.append(
        _check("r_marker_session", MARKER_TAG in tags, MARKER_TAG in tags, "present", False)
    )
    checks.append(_check("r_both_session", BOTH_TAG in tags, BOTH_TAG in tags, "present", False))
    hard = [c["name"] for c in checks if c["hard"] and not c["ok"]]
    soft = [c["name"] for c in checks if not c["hard"] and not c["ok"]]
    return {
        "gate": "I2",
        "decision": "BLOCKED" if hard else ("PASS-WITH-LIMITS" if soft else "PASS"),
        "checks": checks,
        "failed_hard": hard,
        "missed_targets": soft,
        "scope_limits": [
            "no loop closure; reachable-viewpoint partial model (ADR-0008)",
            "R-HARD, out-of-view re-entry and fast motion are outside the current scope",
            "R-MARKER and R-BOTH remain required for complete I2 evidence",
        ],
        "targets_are_provisional": True,
    }


def write_run(
    reports: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    tracking_cfg: TrackingConfig,
    observation_cfg: ObservationConfig,
    root: Path | None = None,
) -> Path:
    hosts = sorted({str(r["host"]) for r in reports if r.get("host")})
    run_id = make_session_id(hosts[0] if len(hosts) == 1 else socket.gethostname())
    directory = (root or paths.data_root()) / "runs" / run_id
    directory.mkdir(parents=True)
    body = {
        "schema_version": SCHEMA_VERSION,
        "type": "TrackingReport",
        "run_id": run_id,
        "sessions": reports,
        "i2": decision,
    }
    files = {"tracking_report.json": atomic_write_json(directory / "tracking_report.json", body)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "P3Run",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "processing_host": socket.gethostname(),
        "software": software_info(),
        "inputs": [
            {
                "session_id": r["session_id"],
                "manifest_sha256": r["manifest_sha256"],
                "mask_revision": r["mask_revision"],
            }
            for r in reports
        ],
        "config": {
            "tracking": {
                "path": tracking_cfg.source_path,
                "sha256": tracking_cfg.source_sha256,
                "values": tracking_cfg.snapshot(),
            },
            "observation": {
                "path": observation_cfg.source_path,
                "sha256": observation_cfg.source_sha256,
            },
        },
        "files": files,
    }
    atomic_write_json(directory / MANIFEST_NAME, manifest)
    return directory
