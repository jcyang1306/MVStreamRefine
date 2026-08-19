import numpy as np

from object_reconstruction.data.models import CameraIntrinsics
from object_reconstruction.geometry.pointcloud import depth_to_pointcloud, transform_points

INTR = CameraIntrinsics(width=8, height=6, fx=100.0, fy=100.0, cx=4.0, cy=3.0)


def test_unproject_principal_point():
    # 1000 mm at the principal point must unproject to (0, 0, 1) meters.
    depth = np.zeros((6, 8), dtype=np.uint16)
    depth[3, 4] = 1000
    points, _ = depth_to_pointcloud(depth, INTR, depth_scale=1000.0)
    assert points.shape == (1, 3)
    assert np.allclose(points[0], [0.0, 0.0, 1.0])


def test_project_unproject_roundtrip():
    depth = np.full((6, 8), 500, dtype=np.uint16)  # 0.5 m everywhere
    points, _ = depth_to_pointcloud(depth, INTR, depth_scale=1000.0)
    # Reproject with K and verify pixel centers are recovered.
    K = INTR.matrix()
    uv = (points @ K.T)
    uv = uv[:, :2] / uv[:, 2:3]
    expected_v, expected_u = np.nonzero(depth > 0)
    assert np.allclose(uv[:, 0], expected_u, atol=1e-9)
    assert np.allclose(uv[:, 1], expected_v, atol=1e-9)


def test_depth_range_and_mask_filters():
    depth = np.zeros((6, 8), dtype=np.uint16)
    depth[0, 0] = 100    # 0.1 m, below min
    depth[1, 1] = 500    # kept
    depth[2, 2] = 3000   # 3 m, above max
    mask = np.zeros((6, 8), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    points, _ = depth_to_pointcloud(
        depth, INTR, depth_scale=1000.0, mask=mask, depth_min_m=0.15, depth_max_m=1.2
    )
    assert points.shape == (1, 3)
    assert np.isclose(points[0, 2], 0.5)


def test_transform_points_identity_and_translation():
    points = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 0.5]])
    assert np.allclose(transform_points(points, np.eye(4)), points)
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(transform_points(points, T), points + [1.0, 2.0, 3.0])
