"""P4 keyframe selection, retrieval and persistence on a recorded test session."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from mvsr.core.errors import ChecksumMismatchError, SessionFormatError
from mvsr.keyframes.benchmark import p4_decision, run_benchmark, write_run
from mvsr.keyframes.config import load_keyframe_config
from mvsr.keyframes.manager import KeyframeManager, _resolve_under
from mvsr.observation.builder import ObservationBuilder
from mvsr.observation.config import load_observation_config
from mvsr.observation.masks import MaskCache
from mvsr.sensors.replay import ReplaySource
from mvsr.tracking.config import load_tracking_config
from mvsr.tracking.session import TrackingSession
from mvsr.tracking.tracker import ObjectTracker
from unit.test_tracking_session import _make


def _run(session: Path, revision: str) -> tuple[KeyframeManager, list[object]]:
    source = ReplaySource(session)
    observation_cfg = load_observation_config()
    masks = MaskCache(session, revision)
    builder = ObservationBuilder(masks, source.bundle, observation_cfg)
    keyframe_cfg = load_keyframe_config()
    keyframe_cfg = replace(
        keyframe_cfg,
        selection=replace(
            keyframe_cfg.selection,
            min_frame_spacing=1,
            max_frame_gap=2,
            min_viewpoint_angle_deg=0.01,
            min_viewpoint_translation_m=0.0001,
        ),
    )
    manager = KeyframeManager(session, masks, keyframe_cfg)
    tracker = ObjectTracker(source.bundle, load_tracking_config(), observation_cfg)
    pipeline = TrackingSession(tracker, keyframe_memory=manager)
    steps = []
    for fid in source.frame_ids:
        frame = source.read(fid)
        steps.append(pipeline.process(frame, builder.process(frame)))
    return manager, steps


def test_keyframes_are_canonical_and_retrieval_is_bounded(
    tmp_path: Path, reference_handeye_path: Path
) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    manager, steps = _run(session, revision)
    assert manager.keyframes
    first = manager.keyframes[0]
    assert first.keyframe_id == 0 and first.pose_version == 0
    assert set(first.raw_refs) == {"color", "depth", "mask"}
    assert first.observation_ref == "observations/000000.npz"
    assert not manager.views[0].observation.keypoints_uv.flags.writeable
    current = next(step for step in reversed(steps) if step.pose.T_cam_obj is not None)
    local_a = manager.local(current.pose.T_cam_obj, 2, manager.views[-1].observation)
    local_b = manager.local(current.pose.T_cam_obj, 2, manager.views[-1].observation)
    assert [view.keyframe_id for view in local_a] == [view.keyframe_id for view in local_b]
    assert len(local_a) <= 2


def test_persistence_roundtrip_and_corruption_detection(
    tmp_path: Path, reference_handeye_path: Path
) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    manager, _ = _run(session, revision)
    directory = manager.persist(tmp_path / "memory")
    loaded = KeyframeManager.load(directory, session, MaskCache(session, revision), manager.cfg)
    assert loaded.digest() == manager.digest()
    assert loaded.T_base_obj_init is not None
    assert manager.T_base_obj_init is not None
    assert loaded.T_base_obj_init.allclose(manager.T_base_obj_init)
    assert [key.checksum for key in loaded.keyframes] == [key.checksum for key in manager.keyframes]
    source = ReplaySource(session)
    tracker = ObjectTracker(source.bundle, load_tracking_config(), load_observation_config())
    tracker.restore(loaded.T_base_obj_init)
    assert tracker.T_base_obj_init.allclose(loaded.T_base_obj_init)
    observation_path = directory / loaded.keyframes[0].observation_ref
    observation_path.write_bytes(observation_path.read_bytes() + b"x")
    with pytest.raises(ChecksumMismatchError, match="sha256"):
        KeyframeManager.load(directory, session, MaskCache(session, revision), manager.cfg)


def test_persistence_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(SessionFormatError, match=r"unsafe|escapes"):
        _resolve_under(tmp_path, "../outside")
    with pytest.raises(SessionFormatError, match=r"unsafe|escapes"):
        _resolve_under(tmp_path, "/absolute")
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "file").write_text("x")
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SessionFormatError, match="escapes"):
        _resolve_under(root, "link/file")


def test_keyframe_choices_are_reproducible(tmp_path: Path, reference_handeye_path: Path) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    first, _ = _run(session, revision)
    second, _ = _run(session, revision)
    assert first.digest() == second.digest()
    assert [key.header.frame_id for key in first.keyframes] == [
        key.header.frame_id for key in second.keyframes
    ]


def test_keyframe_benchmark_persists_and_gates(
    tmp_path: Path, reference_handeye_path: Path
) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    keyframe_cfg = load_keyframe_config()
    tracking_cfg = load_tracking_config()
    observation_cfg = load_observation_config()
    report, manager = run_benchmark(
        session,
        revision,
        keyframe_cfg,
        tracking_cfg,
        observation_cfg,
        limit=6,
    )
    decision: dict[str, Any] = {}
    run = write_run(
        [report],
        [manager],
        decision,
        keyframe_cfg,
        tracking_cfg,
        observation_cfg,
        tmp_path / "runs-root",
    )
    assert (run / "keyframe_report.json").is_file()
    assert report["persistence"]["reload_identical"]
    assert p4_decision([report], keyframe_cfg)["decision"] in (
        "PASS",
        "PASS-WITH-LIMITS",
    )
