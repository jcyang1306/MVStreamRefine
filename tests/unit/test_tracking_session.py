"""P3 tracker/state/window on a hand-built recorded session (control flow, not accuracy evidence)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from mvsr.core.types import (
    KeyframeView,
    MapView,
    PoseStatus,
    ReasonCode,
    TrackingReferences,
)
from mvsr.observation.builder import ObservationBuilder
from mvsr.observation.config import load_observation_config
from mvsr.observation.masks import MaskCache, MaskCacheWriter, MaskStatus, mask_revision_id
from mvsr.sensors.replay import ReplaySource
from mvsr.tracking.benchmark import i2_decision, run_benchmark, write_run
from mvsr.tracking.config import load_tracking_config
from mvsr.tracking.session import TrackingSession, pose_digest
from mvsr.tracking.tracker import ObjectTracker, TrackingError
from unit.test_replay import make_session

H, W = 120, 160


def _images(i: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(91)
    base = rng.integers(0, 256, (H // 3 + 6, W // 3 + 6), dtype=np.uint8)
    texture = cv2.resize(base, (W + 20, H + 20), interpolation=cv2.INTER_NEAREST)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    gray = np.ascontiguousarray(texture[10 : 10 + H, 10 - i : 10 - i + W])
    depth = np.full((H, W), 4000, dtype=np.uint16)
    return np.stack([gray, gray, gray], axis=2), depth


def _make(tmp_path: Path, handeye: Path) -> tuple[Path, str]:
    session = make_session(
        tmp_path,
        handeye,
        n=7,
        gt_type="constant_T_base_obj",
        size=(W, H),
        images=_images,
        dropped=(),
    )
    mask = np.zeros((H, W), dtype=bool)
    mask[15:105, 20:140] = True
    revision = mask_revision_id("test", {"case": "tracking"})
    writer = MaskCacheWriter(session, "wrist", revision, {})
    for fid in ReplaySource(session).frame_ids:
        if fid == 3:
            writer.add(fid, np.zeros_like(mask), MaskStatus.EMPTY, None)
        else:
            writer.add(fid, mask, MaskStatus.OK, 0.99)
    writer.finalize()
    return session, revision


def _run(session: Path, revision: str):  # type: ignore[no-untyped-def]
    source = ReplaySource(session)
    ocfg = load_observation_config()
    builder = ObservationBuilder(MaskCache(session, revision), source.bundle, ocfg)
    tracker = ObjectTracker(source.bundle, load_tracking_config(), ocfg)
    pipeline = TrackingSession(tracker)
    steps = []
    for fid in source.frame_ids:
        frame = source.read(fid)
        steps.append(pipeline.process(frame, builder.process(frame)))
    return tracker, pipeline, steps


def test_tracking_initializes_gauge_and_handles_empty_mask(
    tmp_path: Path, reference_handeye_path: Path
) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    tracker, pipeline, steps = _run(session, revision)
    first = steps[0]
    assert first.pose.status is PoseStatus.VALID
    assert first.pose.source == "gauge_initialization"
    assert tracker.T_base_obj_init is not None
    assert first.pose.T_base_obj == tracker.T_base_obj_init
    assert first.keyframe_inserted == 0
    empty = next(s for s in steps if s.pose.header.frame_id == 3)
    assert empty.pose.status is PoseStatus.INVALID
    assert ReasonCode.MASK_TOO_SMALL in empty.pose.reason_codes
    assert len(pipeline.window.all) >= 1


def test_tracking_replay_is_deterministic(tmp_path: Path, reference_handeye_path: Path) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    a = [s.pose for s in _run(session, revision)[2]]
    b = [s.pose for s in _run(session, revision)[2]]
    assert pose_digest(a) == pose_digest(b)
    assert any(p.status is not PoseStatus.INVALID for p in a[1:])


def test_tracker_accepts_equivalent_header_and_reset_restarts(
    tmp_path: Path, reference_handeye_path: Path
) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    source = ReplaySource(session)
    config = load_observation_config()
    builder = ObservationBuilder(MaskCache(session, revision), source.bundle, config)
    frame = source.read(0)
    observation = builder.process(frame)
    equivalent = replace(observation, header=replace(observation.header))
    tracker = ObjectTracker(source.bundle, load_tracking_config(), config)
    pose = tracker.track(frame, equivalent, TrackingReferences(None, None, ()))
    assert pose.status is PoseStatus.VALID
    with pytest.raises(TrackingError, match="increase"):
        tracker.track(frame, equivalent, TrackingReferences(pose, None, ()))
    tracker.reset()
    assert (
        tracker.track(frame, equivalent, TrackingReferences(None, None, ())).status
        is PoseStatus.VALID
    )


def test_map_fallback_and_stale_version(tmp_path: Path, reference_handeye_path: Path) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    source = ReplaySource(session)
    config = load_observation_config()
    builder = ObservationBuilder(MaskCache(session, revision), source.bundle, config)
    tracker = ObjectTracker(source.bundle, load_tracking_config(), config)
    frame0 = source.read(0)
    obs0 = builder.process(frame0)
    first = tracker.track(frame0, obs0, TrackingReferences(None, None, ()))
    assert first.T_cam_obj is not None
    T_obj_cam = first.T_cam_obj.inverse()
    assert obs0.normals_cam is not None
    map_view = MapView(
        map_version=1,
        points_obj_m=T_obj_cam.transform_points(obs0.points_cam_m).astype(np.float32),
        normals_obj=(obs0.normals_cam @ T_obj_cam.R.T).astype(np.float32),
        colors_u8=None,
        source_keyframes={},
    )
    frame1 = source.read(1)
    obs1 = builder.process(frame1)
    predicted = tracker.predict(frame1, first)
    fallback = tracker.track(frame1, obs1, TrackingReferences(first, predicted, ()), map_view)
    assert fallback.status is not PoseStatus.INVALID

    tracker.reset()
    first = tracker.track(frame0, obs0, TrackingReferences(None, None, ()))
    assert first.T_cam_obj is not None
    keyframe = KeyframeView(0, obs0, first.T_cam_obj, 0)
    stale_map = replace(map_view, source_keyframes={0: 1})
    stale = tracker.track(
        frame1,
        obs1,
        TrackingReferences(first, tracker.predict(frame1, first), (keyframe,)),
        stale_map,
    )
    assert stale.status is PoseStatus.INVALID
    assert stale.reason_codes == (ReasonCode.STALE_VERSION,)


def test_tracking_benchmark_and_i2_partial_gate(
    tmp_path: Path, reference_handeye_path: Path
) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    cfg = load_tracking_config()
    cfg = replace(cfg, gate=replace(cfg.gate, min_valid_ratio=0.1))
    observation_cfg = load_observation_config()
    report = run_benchmark(session, revision, cfg, observation_cfg, limit=6)
    assert report["reproducibility"]["identical"]
    assert report["keyframes"]["inserted"] >= 1
    decision = i2_decision([report], cfg)
    assert decision["decision"] == "PASS-WITH-LIMITS"
    assert decision["failed_hard"] == []
    assert {"r_marker_session", "r_both_session"} <= set(decision["missed_targets"])
    run = write_run([report], decision, cfg, observation_cfg, tmp_path)
    assert (run / "tracking_report.json").is_file()
