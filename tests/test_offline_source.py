from pathlib import Path

import numpy as np
import pytest

from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def source():
    config = load_config(REPO_ROOT / "configs" / "offline.yaml")
    config["dataset"]["root"] = str(REPO_ROOT / "data")
    return OfflineFrameSource.from_config(config)


def test_frame_count_and_completeness(source):
    assert len(source) == 113
    assert source.indices[0] == 0
    assert source.indices[-1] == 112
    assert source.missing_files == {}


def test_first_packet_contents(source):
    packet = source.read_packet(0)
    assert packet.index == 0
    assert packet.timestamp is None
    for cam in (packet.cam1, packet.cam2):
        assert cam.rgb.shape == (720, 1280, 3)
        assert cam.rgb.dtype == np.uint8
        assert cam.depth.shape == (720, 1280)
        assert cam.depth.dtype == np.uint16
    assert np.allclose(packet.T_world_cam1, np.eye(4))


def test_confirmed_conventions_produce_world_pose(source):
    assert source.conventions_confirmed is True
    packet = source.read_packet(0)
    T = packet.T_world_cam2
    assert T is not None and T.shape == (4, 4)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-6
    assert np.allclose(T[3], [0, 0, 0, 1])
    # Composition matches the confirmed formula.
    from object_reconstruction.calibration.transforms import (
        compose_T_world_cam2,
        pose7d_to_matrix,
    )

    expected = compose_T_world_cam2(
        source.T_base_cam1, pose7d_to_matrix(packet.raw_pose_7d), source.T_tcp_cam2
    )
    assert np.allclose(T, expected)
    # Raw pose is still carried through.
    expected_first_row = np.array(
        [0.19252, -0.202075, 0.262422, 0.799332013, 0.108328487, -0.530382538, 0.260821078]
    )
    assert np.allclose(packet.raw_pose_7d, expected_first_row)


def test_unconfirmed_conventions_block_world_pose():
    config = load_config(REPO_ROOT / "configs" / "offline.yaml")
    config["dataset"]["root"] = str(REPO_ROOT / "data")
    config["dataset"]["pose_semantics"] = None
    blocked = OfflineFrameSource.from_config(config)
    assert blocked.conventions_confirmed is False
    packet = blocked.read_packet(0)
    assert packet.T_world_cam2 is None


def test_pose_table_matches_per_frame_files(source):
    for position in (0, 56, 112):
        per_frame = source.load_per_frame_pose(source.indices[position])
        assert np.abs(per_frame - source.raw_poses[position]).max() < 1e-4
