"""Depth preprocessing (P2 O2) and ray grids on hand-written depth images."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from mvsr.calib.bundle import build_bundle
from mvsr.calib.handeye import load_handeye
from mvsr.calib.intrinsics import ray_grid, rays_for_pixels, undistort_normalized
from mvsr.core.depth import depth_raw_to_m
from mvsr.observation.config import load_observation_config
from mvsr.observation.depth import (
    depth_geometry,
    depth_sigma_m,
    sample_depth,
    silhouette_offsets_px,
)

H, W = 60, 80


def _bundle(handeye: Path, coeffs: list[float] | None = None):  # type: ignore[no-untyped-def]
    return build_bundle(
        camera_id="wrist",
        camera_serial="323622271240",
        camera_model="Intel RealSense D405",
        width=W,
        height=H,
        fx=70.0,
        fy=70.0,
        cx=39.5,
        cy=29.5,
        distortion_model="inverse_brown_conrady",
        distortion=coeffs or [0.0] * 5,
        depth_units_m=1e-4,
        handeye=load_handeye(handeye),
    )


def _geom(raw: np.ndarray, mask: np.ndarray, handeye: Path, **cfg_kw: object):  # type: ignore[no-untyped-def]
    cfg = load_observation_config().depth
    cfg = replace(cfg, **cfg_kw)  # type: ignore[arg-type]
    z = depth_raw_to_m(raw, 1e-4)
    return depth_geometry(z, raw != 0, mask, _bundle(handeye), cfg), z


def test_ray_grid_matches_pointwise_undistortion(reference_handeye_path: Path) -> None:
    b = _bundle(reference_handeye_path, [-0.05, 0.06, 0.0009, 0.0008, -0.02])
    gx, gy = ray_grid(b)
    assert gx.shape == (H, W) and ray_grid(b)[0] is gx  # cached per calibration_id
    n = undistort_normalized(
        np.array([[0.0, 0.0], [79.0, 59.0], [13.0, 41.0]]), b.K, b.distortion_model, b.distortion
    )
    np.testing.assert_allclose([gx[0, 0], gx[59, 79], gx[41, 13]], n[:, 0], atol=1e-6)
    np.testing.assert_allclose([gy[0, 0], gy[59, 79], gy[41, 13]], n[:, 1], atol=1e-6)
    np.testing.assert_allclose(rays_for_pixels(b, [[13.0, 41.0]])[0], [n[2, 0], n[2, 1], 1.0])


def test_plane_gives_coplanar_points_and_camera_facing_normals(
    reference_handeye_path: Path,
) -> None:
    raw = np.full((H, W), 4000, np.uint16)
    mask = np.zeros((H, W), bool)
    mask[10:50, 15:65] = True
    g, _z = _geom(raw, mask, reference_handeye_path)
    assert g.geometry_mask.sum() == (40 - 6) * (50 - 6)  # eroded by 3 px
    assert not (g.geometry_mask & ~mask).any()
    np.testing.assert_allclose(g.points_cam_m[:, 2], 0.4, atol=1e-6)
    np.testing.assert_allclose(
        g.normals_cam, np.tile([0.0, 0.0, -1.0], (g.normals_cam.shape[0], 1)), atol=1e-6
    )
    u, v = g.pixel_uv[0]
    gx, gy = ray_grid(_bundle(reference_handeye_path))
    np.testing.assert_allclose(g.points_cam_m[0], [gx[v, u] * 0.4, gy[v, u] * 0.4, 0.4], atol=1e-6)
    assert g.stats["valid_depth_ratio"] == 1.0 and g.stats["removed_erosion_px"] == mask.sum() - (
        34 * 44
    )


def test_steps_holes_range_and_noise_are_removed_not_filled(reference_handeye_path: Path) -> None:
    raw = np.full((H, W), 4000, np.uint16)
    raw[:, 40:] = 5000  # step: 10 cm jump
    raw[20:23, 20:23] = 0  # hole
    raw[40:44, 20:24] = 7000  # beyond max_depth 0.6 m
    raw[30, 50] = 5600  # spike on the far side
    mask = np.ones((H, W), bool)
    g, z = _geom(raw, mask, reference_handeye_path, mask_erode_px=0)
    gm = g.geometry_mask
    assert not gm[:, 39:41].any()  # both sides of the step
    ring = np.zeros_like(gm)
    ring[19:24, 20:23] = True
    ring[20:23, 19:24] = True  # hole + its 4-neighbours (diagonals stay: 4-neighbour rule)
    assert not gm[ring].any() and gm[19, 19] and gm[17, 17] and gm[25, 25]
    assert not gm[40:44, 20:24].any()
    assert not gm[29:32, 49:52].any() and gm[30, 55]
    assert np.all((z[gm] > 0) & (z[gm] <= 0.6))
    assert g.stats["removed_edge_px"] > 0


def test_noise_threshold_and_sampling(reference_handeye_path: Path) -> None:
    rng = np.random.default_rng(3)
    raw = (4000 + rng.normal(0, 60, (H, W))).astype(np.uint16)  # 6 mm std at 0.4 m >> sigma 2.9 mm
    mask = np.ones((H, W), bool)
    g, z = _geom(raw, mask, reference_handeye_path, mask_erode_px=0, edge_jump_rel=0.5, noise_k=1.0)
    assert g.geometry_mask.mean() < 0.5
    assert abs(float(depth_sigma_m(0.4, load_observation_config().depth)) - 0.00292) < 1e-6
    d = sample_depth(z, g.geometry_mask, np.array([[10.2, 10.4], [-3.0, 2.0], [79.4, 59.4]]))
    assert d.dtype == np.float32 and np.isnan(d[1])
    assert np.isnan(d[0]) == (not g.geometry_mask[10, 10])


def test_silhouette_offset_measures_depth_to_colour_registration() -> None:
    v, u = np.mgrid[0:H, 0:W]
    disc = (u - 40) ** 2 + (v - 30) ** 2 <= 18**2
    raw = np.where(disc, 4000, 5000).astype(np.uint16)
    z = depth_raw_to_m(raw, 1e-4)
    exact = silhouette_offsets_px(disc, z, raw != 0, 0.02)
    assert exact.size > 10 and abs(float(np.median(exact))) <= 0.5
    small = (u - 40) ** 2 + (v - 30) ** 2 <= 15**2  # colour mask 3 px inside the depth step
    off = silhouette_offsets_px(small, z, raw != 0, 0.02)
    assert 2.0 <= float(np.median(off)) <= 4.0
    flat = depth_raw_to_m(np.full((H, W), 4000, np.uint16), 1e-4)
    assert silhouette_offsets_px(disc, flat, raw != 0, 0.02).size == 0
    assert silhouette_offsets_px(np.zeros((H, W), bool), z, raw != 0, 0.02).size == 0
