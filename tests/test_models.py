from pathlib import Path

import numpy as np
import pytest

from object_reconstruction.calibration.handeye import load_handeye_matrices
from object_reconstruction.calibration.intrinsics import (
    intrinsics_from_matrix,
    load_intrinsics_matrix,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Golden values from PLAN section 5.1 (parsed from data/intrinsic/*.txt).
GOLDEN = {
    "head": (908.781982421875, 908.7359619140625, 648.3536376953125, 370.1097106933594),
    "wrist": (653.2439575195312, 652.7537231445312, 636.3500366210938, 358.8265380859375),
}


@pytest.mark.parametrize("prefix", ["head", "wrist"])
def test_intrinsics_golden_values(prefix):
    K = load_intrinsics_matrix(DATA_ROOT / "intrinsic" / f"{prefix}_cam_K.txt")
    intr = intrinsics_from_matrix(K, width=1280, height=720)
    fx, fy, cx, cy = GOLDEN[prefix]
    assert intr.fx == pytest.approx(fx)
    assert intr.fy == pytest.approx(fy)
    assert intr.cx == pytest.approx(cx)
    assert intr.cy == pytest.approx(cy)
    assert np.allclose(intr.matrix(), K)


def test_intrinsics_rejects_bad_principal_point():
    K = np.array([[900.0, 0, 5000.0], [0, 900.0, 360.0], [0, 0, 1]])
    with pytest.raises(ValueError):
        intrinsics_from_matrix(K, width=1280, height=720)


def test_handeye_matrices_are_rigid():
    matrices = load_handeye_matrices(DATA_ROOT / "handeye" / "handeye_tf.txt")
    assert set(matrices) == {"wrist_cam2", "base_cam1"}
    for T in matrices.values():
        assert T.shape == (4, 4)
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-3
        assert np.allclose(T[3], [0, 0, 0, 1])
