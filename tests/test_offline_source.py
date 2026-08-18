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


def test_conventions_unconfirmed_blocks_world_pose(source):
    assert source.conventions_confirmed is False
    packet = source.read_packet(0)
    assert packet.T_world_cam2 is None
    # Raw pose is still carried through for inspection.
    expected_first_row = np.array(
        [0.19252, -0.202075, 0.262422, 0.799332013, 0.108328487, -0.530382538, 0.260821078]
    )
    assert np.allclose(packet.raw_pose_7d, expected_first_row)


def test_pose_table_matches_per_frame_files(source):
    for position in (0, 56, 112):
        per_frame = source.load_per_frame_pose(source.indices[position])
        assert np.abs(per_frame - source.raw_poses[position]).max() < 1e-4
