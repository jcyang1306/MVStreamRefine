"""P4 keyframe-memory benchmark, stage gate and durable run artifacts."""

from __future__ import annotations

import socket
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from mvsr import paths
from mvsr.core.codec import SCHEMA_VERSION
from mvsr.core.types import PoseStatus, make_session_id
from mvsr.io.atomic import atomic_write_json, sha256_file
from mvsr.io.session_writer import MANIFEST_NAME
from mvsr.observation.builder import ObservationBuilder
from mvsr.observation.config import ObservationConfig
from mvsr.observation.masks import MaskCache
from mvsr.sensors.config import SensorsConfig
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.stats import summary
from mvsr.software import software_info
from mvsr.tracking.benchmark import STATIC_GT, _spread
from mvsr.tracking.config import TrackingConfig
from mvsr.tracking.session import TrackingSession, pose_digest
from mvsr.tracking.tracker import ObjectTracker

from .config import KeyframeConfig
from .manager import KeyframeManager, memory_revision_id


def _run_once(
    session_dir: str | Path,
    revision: str,
    keyframe_cfg: KeyframeConfig,
    tracking_cfg: TrackingConfig,
    observation_cfg: ObservationConfig,
    sensors_cfg: SensorsConfig | None,
    limit: int | None,
) -> tuple[dict[str, Any], KeyframeManager]:
    source = ReplaySource(session_dir, "wrist", config=sensors_cfg)
    masks = MaskCache(source.reader.dir, revision, "wrist")
    builder = ObservationBuilder(masks, source.bundle, observation_cfg)
    manager = KeyframeManager(source.reader.dir, masks, keyframe_cfg)
    tracker = ObjectTracker(source.bundle, tracking_cfg, observation_cfg)
    pipeline = TrackingSession(tracker, keyframe_memory=manager)
    ids = [fid for fid in source.frame_ids if fid in masks][:limit]
    poses = []
    decisions: Counter[str] = Counter()
    retrieval_recall: list[float] = []
    local_sizes: list[float] = []
    insert_frames: list[int] = []
    novelty: dict[str, list[float]] = {
        "translation_m": [],
        "angle_deg": [],
        "appearance_hamming": [],
        "covisibility": [],
        "spatial_novelty": [],
    }
    immutable = True
    started = time.monotonic()
    for fid in ids:
        frame = source.read(fid)
        observation = builder.process(frame)
        step = pipeline.process(frame, observation)
        poses.append(step.pose)
        decision = str(manager.last_decision.get("decision", "NONE"))
        decisions[decision] += 1
        if step.keyframe_inserted is not None:
            insert_frames.append(fid)
            for key, value in (manager.last_decision.get("metrics") or {}).items():
                if key in novelty:
                    novelty[key].append(float(value))
        recall = manager.last_retrieval.get("recall")
        if recall is not None:
            retrieval_recall.append(float(recall))
        local_sizes.append(float(len(step.local_keyframe_ids)))
        immutable = immutable and (
            not observation.keypoints_uv.flags.writeable
            and not observation.points_cam_m.flags.writeable
            and not observation.descriptors.flags.writeable
        )
    elapsed = time.monotonic() - started
    scenario = source.reader.manifest.get("scenario") or {}
    base_poses = [
        pose.T_base_obj
        for pose in poses
        if pose.status is not PoseStatus.INVALID and pose.T_base_obj is not None
    ]
    gaps = [float(b - a) for a, b in pairwise(insert_frames)]
    revision_params = {
        "session_manifest_sha256": sha256_file(Path(source.reader.dir) / MANIFEST_NAME),
        "mask_revision": revision,
        "keyframe_config_sha256": keyframe_cfg.source_sha256,
        "tracking_config_sha256": tracking_cfg.source_sha256,
        "observation_config_sha256": observation_cfg.source_sha256,
    }
    report = {
        "session_id": source.session_id,
        "host": source.reader.manifest.get("host"),
        "manifest_sha256": revision_params["session_manifest_sha256"],
        "scenario": scenario,
        "mask_revision": revision,
        "memory_revision": memory_revision_id(revision_params),
        "frames": {
            "processed": len(poses),
            "valid": sum(p.status is PoseStatus.VALID for p in poses),
        },
        "keyframes": {
            "count": len(manager.keyframes),
            "frame_ids": [key.header.frame_id for key in manager.keyframes],
            "insert_gap_frames": summary(gaps),
            "decisions": dict(sorted(decisions.items())),
            "revisit_reuses": manager.revisit_reuses,
            "revisit_opportunities": manager.revisit_opportunities,
            "capacity_rejections": manager.capacity_rejections,
            "digest": manager.digest(),
        },
        "novelty_at_insert": {key: summary(values) for key, values in novelty.items()},
        "retrieval": {
            "best_anchor_hit_rate": summary(retrieval_recall),
            "local_size": summary(local_sizes),
            "queries": len(retrieval_recall),
        },
        "revisit_reuse_ratio": (
            manager.revisit_reuses / manager.revisit_opportunities
            if manager.revisit_opportunities
            else None
        ),
        "memory_bound_ok": len(manager.keyframes) <= keyframe_cfg.retrieval.max_global_keyframes,
        "immutable_observations": immutable,
        "pose_digest": pose_digest(poses),
        "static_spread": _spread(base_poses, tracker.T_base_obj_init)
        if scenario.get("gt_type") == STATIC_GT
        else None,
        "runtime": {
            "wall_s": round(elapsed, 3),
            "fps": round(len(poses) / elapsed, 3) if elapsed else None,
        },
        "config_sha256": keyframe_cfg.source_sha256,
        "tracking_config_sha256": tracking_cfg.source_sha256,
        "observation_config_sha256": observation_cfg.source_sha256,
    }
    return report, manager


def run_benchmark(
    session_dir: str | Path,
    revision: str,
    keyframe_cfg: KeyframeConfig,
    tracking_cfg: TrackingConfig,
    observation_cfg: ObservationConfig,
    sensors_cfg: SensorsConfig | None = None,
    *,
    limit: int | None = None,
) -> tuple[dict[str, Any], KeyframeManager]:
    report, manager = _run_once(
        session_dir,
        revision,
        keyframe_cfg,
        tracking_cfg,
        observation_cfg,
        sensors_cfg,
        limit,
    )
    n = min(
        report["frames"]["processed"],
        keyframe_cfg.benchmark.reproducibility_frames,
    )
    first, _ = _run_once(
        session_dir,
        revision,
        keyframe_cfg,
        tracking_cfg,
        observation_cfg,
        sensors_cfg,
        n,
    )
    second, _ = _run_once(
        session_dir,
        revision,
        keyframe_cfg,
        tracking_cfg,
        observation_cfg,
        sensors_cfg,
        n,
    )
    report["reproducibility"] = {
        "frames": n,
        "identical": (
            first["keyframes"]["digest"] == second["keyframes"]["digest"]
            and first["keyframes"]["frame_ids"] == second["keyframes"]["frame_ids"]
            and first["pose_digest"] == second["pose_digest"]
        ),
        "memory_digest": first["keyframes"]["digest"],
        "pose_digest": first["pose_digest"],
    }
    return report, manager


def _check(name: str, ok: bool, value: Any, target: str, hard: bool) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "value": value, "target": target, "hard": hard}


def p4_decision(reports: list[dict[str, Any]], cfg: KeyframeConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [_check("sessions", bool(reports), len(reports), ">= 1", True)]
    for report in reports:
        sid = report["session_id"]
        count = report["keyframes"]["count"]
        checks.extend(
            [
                _check(
                    f"{sid}/keyframes",
                    0 < count <= cfg.gate.max_keyframes,
                    count,
                    f"1..{cfg.gate.max_keyframes}",
                    True,
                ),
                _check(
                    f"{sid}/reproducible",
                    report["reproducibility"]["identical"],
                    report["reproducibility"]["frames"],
                    "identical choices and digests",
                    True,
                ),
                _check(
                    f"{sid}/immutable",
                    report["immutable_observations"],
                    report["immutable_observations"],
                    "true",
                    True,
                ),
            ]
        )
        hit_rate = (report["retrieval"]["best_anchor_hit_rate"] or {}).get("median")
        checks.append(
            _check(
                f"{sid}/best_anchor_hit_rate",
                hit_rate is not None and hit_rate >= cfg.gate.min_best_anchor_hit_rate,
                hit_rate,
                f">= {cfg.gate.min_best_anchor_hit_rate}",
                False,
            )
        )
        reuse = report["revisit_reuse_ratio"]
        checks.append(
            _check(
                f"{sid}/revisit_reuse_ratio",
                reuse is None or reuse >= cfg.gate.min_revisit_reuse_ratio,
                reuse,
                f">= {cfg.gate.min_revisit_reuse_ratio} when revisits exist",
                False,
            )
        )
        persistence = report.get("persistence") or {}
        checks.append(
            _check(
                f"{sid}/persistence_reload",
                bool(persistence.get("reload_identical")),
                persistence.get("reload_identical"),
                "true",
                True,
            )
        )
    hard = [c["name"] for c in checks if c["hard"] and not c["ok"]]
    soft = [c["name"] for c in checks if not c["hard"] and not c["ok"]]
    return {
        "gate": "P4",
        "decision": "BLOCKED" if hard else ("PASS-WITH-LIMITS" if soft else "PASS"),
        "checks": checks,
        "failed_hard": hard,
        "missed_targets": soft,
        "scope_limits": [
            "keyframes cover only the reachable viewpoint arc (ADR-0008)",
            "this is the P4 stage gate; I3 additionally requires P5 mapping",
        ],
        "targets_are_provisional": True,
    }


def write_run(
    reports: list[dict[str, Any]],
    managers: list[KeyframeManager],
    decision: dict[str, Any] | None,
    keyframe_cfg: KeyframeConfig,
    tracking_cfg: TrackingConfig,
    observation_cfg: ObservationConfig,
    root: Path | None = None,
) -> Path:
    hosts = sorted({str(r["host"]) for r in reports if r.get("host")})
    run_id = make_session_id(hosts[0] if len(hosts) == 1 else socket.gethostname())
    directory = (root or paths.data_root()) / "runs" / run_id
    directory.mkdir(parents=True)
    for report, manager in zip(reports, managers, strict=True):
        memory_dir = directory / "keyframes" / report["session_id"]
        manager.persist(memory_dir)
        loaded = KeyframeManager.load(
            memory_dir,
            manager.session_dir,
            manager.masks,
            manager.cfg,
        )
        report["persistence"] = {
            "path": str(memory_dir.relative_to(directory)),
            "reload_identical": loaded.digest() == manager.digest(),
            "memory_digest": loaded.digest(),
        }
    if decision is not None:
        decision.clear()
        decision.update(p4_decision(reports, keyframe_cfg))
    body = {
        "schema_version": SCHEMA_VERSION,
        "type": "KeyframeReport",
        "run_id": run_id,
        "sessions": reports,
        "p4": decision,
    }
    atomic_write_json(directory / "keyframe_report.json", body)
    files = {
        str(path.relative_to(directory)): sha256_file(path)
        for path in directory.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "P4Run",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "processing_host": socket.gethostname(),
        "software": software_info(),
        "inputs": [
            {
                "session_id": report["session_id"],
                "manifest_sha256": report["manifest_sha256"],
                "mask_revision": report["mask_revision"],
            }
            for report in reports
        ],
        "config": {
            "keyframes": {
                "path": keyframe_cfg.source_path,
                "sha256": keyframe_cfg.source_sha256,
                "values": keyframe_cfg.snapshot(),
            },
            "tracking": {
                "path": tracking_cfg.source_path,
                "sha256": tracking_cfg.source_sha256,
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
