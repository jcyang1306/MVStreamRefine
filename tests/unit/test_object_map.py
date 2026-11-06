"""P5 object-frame TSDF and conservative commit tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mvsr.core.types import Keyframe, PoseStatus
from mvsr.keyframes.manager import KeyframeManager
from mvsr.mapping.config import load_mapping_config
from mvsr.mapping.map import MappingError, ObjectMap, pose_from_keyframe
from mvsr.sensors.replay import ReplaySource
from unit.test_keyframe_manager import _run
from unit.test_tracking_session import _make


def _object_map(
    tmp_path: Path, handeye: Path
) -> tuple[ObjectMap, KeyframeManager, tuple[Keyframe, ...]]:
    session, revision = _make(tmp_path, handeye)
    manager, _ = _run(session, revision)
    assert manager.T_base_obj_init is not None
    source = ReplaySource(session)
    object_map = ObjectMap(
        session,
        source.bundle,
        manager,
        manager.T_base_obj_init,
        load_mapping_config(),
    )
    return object_map, manager, manager.keyframes


def test_provisional_tsdf_is_in_object_frame(tmp_path: Path, reference_handeye_path: Path) -> None:
    object_map, manager, keyframes = _object_map(tmp_path, reference_handeye_path)
    geometry = object_map.build_provisional(list(keyframes[:2]))
    assert len(geometry.points_obj_m) > 100
    first = keyframes[0]
    observation = manager.observation(first.keyframe_id)
    expected = first.T_cam_obj_current.inverse().transform_points(observation.points_cam_m)
    np.testing.assert_allclose(
        np.median(geometry.points_obj_m, axis=0),
        np.median(expected, axis=0),
        atol=0.015,
    )
    from mvsr.core.se3 import PoseSE3
    from mvsr.mapping.open3d_backend import create_volume, extract, integrate

    inverted = create_volume(object_map.cfg.tsdf)
    raw = manager.reader.read_frame("wrist", first.header.frame_id)
    integrate(
        inverted,
        observation,
        raw,
        object_map.bundle,
        PoseSE3(first.T_cam_obj_current.inverse().matrix, "cam", "obj"),
        object_map.cfg.tsdf,
    )
    wrong = extract(inverted, color=False)
    correct_error = np.linalg.norm(
        np.median(geometry.points_obj_m, axis=0) - np.median(expected, axis=0)
    )
    wrong_error = np.linalg.norm(
        np.median(wrong.points_obj_m, axis=0) - np.median(expected, axis=0)
    )
    assert wrong_error > 2 * max(correct_error, 0.001)
    assert object_map.map_version == 0
    with pytest.raises(MappingError, match="published"):
        object_map.snapshot()


def test_commit_is_versioned_idempotent_and_rejects_bad_pose(
    tmp_path: Path, reference_handeye_path: Path
) -> None:
    object_map, _manager, keyframes = _object_map(tmp_path, reference_handeye_path)
    keyframe = keyframes[0]
    pose = pose_from_keyframe(keyframe)
    assert object_map.commit(keyframe, pose) == 1
    assert object_map.commit(keyframe, pose) == 1
    view = object_map.view()
    assert view.map_version == 1 and view.source_keyframes == {keyframe.keyframe_id: 0}
    degraded = replace(pose, status=PoseStatus.DEGRADED)
    with pytest.raises(MappingError, match="VALID"):
        object_map.commit(keyframes[-1], degraded)
    stale = replace(pose_from_keyframe(keyframes[-1]), pose_version=1)
    with pytest.raises(MappingError, match="STALE_VERSION"):
        object_map.commit(keyframes[-1], stale)


def test_publish_writes_snapshot_and_geometry(tmp_path: Path, reference_handeye_path: Path) -> None:
    object_map, _manager, keyframes = _object_map(tmp_path, reference_handeye_path)
    object_map.commit(keyframes[0], pose_from_keyframe(keyframes[0]))
    snapshot = object_map.publish(tmp_path / "published", "20261106T120000_CW20175_abcd")
    assert snapshot.map_version == 1
    assert snapshot.T_base_obj_init == object_map.T_base_obj_init
    assert (tmp_path / "published" / "snapshot.json").is_file()
    assert object_map.snapshot() == snapshot
