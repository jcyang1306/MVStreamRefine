"""Correspondence screening on a known rigid motion with planted outliers (P2 O5)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from mvsr.calib.bundle import build_bundle
from mvsr.calib.handeye import load_handeye
from mvsr.calib.intrinsics import project_cam_points
from mvsr.core.se3 import so3_exp
from mvsr.observation.config import load_observation_config
from mvsr.observation.correspondence import KeypointView, kabsch, match_observations

CFG = load_observation_config()


def _bundle(handeye: Path):  # type: ignore[no-untyped-def]
    return build_bundle(
        camera_id="wrist",
        camera_serial="323622271240",
        camera_model="Intel RealSense D405",
        width=1280,
        height=720,
        fx=650.0,
        fy=650.0,
        cx=640.0,
        cy=360.0,
        distortion_model="inverse_brown_conrady",
        distortion=[0.0] * 5,
        depth_units_m=1e-4,
        handeye=load_handeye(handeye),
    )


def _scene(n: int = 60, outliers: int = 12):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(5)
    P = np.column_stack(
        [rng.uniform(-0.05, 0.05, n), rng.uniform(-0.04, 0.04, n), rng.uniform(0.33, 0.42, n)]
    )
    R, t = so3_exp([0.02, -0.05, 0.03]), np.array([0.01, -0.004, 0.006])
    Q = P @ R.T + t
    K = np.array([[650.0, 0, 640.0], [0, 650.0, 360.0], [0, 0, 1]])
    uv_s, uv_t = project_cam_points(P, K), project_cam_points(Q, K)
    bad = np.arange(n - outliers, n)
    uv_t[bad] += rng.uniform(25, 60, (outliers, 2)) * rng.choice([-1, 1], (outliers, 2))
    zt = Q[:, 2].copy()
    zt[bad] += 0.03
    return P, Q, uv_s, uv_t, zt, bad, R, t


def _views(uv_s, uv_t, zs, zt, ids_t=None, desc=None):  # type: ignore[no-untyped-def]
    n = uv_s.shape[0]
    ids = np.arange(n, dtype=np.int64)
    d = desc if desc is not None else np.zeros((n, 32), np.uint8)
    a = KeypointView(0, uv_s.astype(np.float32), zs.astype(np.float32), ids, d, "orb")
    b = KeypointView(
        5, uv_t.astype(np.float32), zt.astype(np.float32), ids if ids_t is None else ids_t, d, "orb"
    )
    return a, b


def test_kabsch_recovers_exact_motion() -> None:
    P, Q, *_, R, t = _scene(outliers=0)
    R2, t2 = kabsch(P, Q)
    np.testing.assert_allclose(R2, R, atol=1e-9)
    np.testing.assert_allclose(t2, t, atol=1e-9)


def test_rigid_screening_flags_planted_outliers(reference_handeye_path: Path) -> None:
    P, _Q, uv_s, uv_t, zt, bad, _R, t = _scene()
    a, b = _views(uv_s, uv_t, P[:, 2], zt)
    cs, diag = match_observations(
        a, b, _bundle(reference_handeye_path), CFG.correspondence, CFG.depth, "track"
    )
    assert diag.model == "rigid" and diag.candidates == 60
    expected = np.ones(60, bool)
    expected[bad] = False
    np.testing.assert_array_equal(cs.inlier, expected)
    assert {cs.rejection_reason[k] for k in bad} == {"RIGID_RESIDUAL"}
    np.testing.assert_allclose(diag.rigid_t, t, atol=1e-3)
    assert np.all(np.isfinite(cs.sigma_m)) and cs.method == "track+rigid"
    again, _ = match_observations(
        a, b, _bundle(reference_handeye_path), CFG.correspondence, CFG.depth, "track"
    )
    np.testing.assert_array_equal(again.inlier, cs.inlier)


def test_one_sided_and_missing_depth_use_the_model(reference_handeye_path: Path) -> None:
    P, _Q, uv_s, uv_t, zt, bad, _R, _t = _scene()
    zt = zt.copy()
    zt[40:48] = np.nan  # good matches, target depth missing -> reprojection check
    zt[bad[:4]] = np.nan  # outliers, target depth missing -> reprojection check fails
    zs = P[:, 2].copy()
    zs[30:34] = np.nan
    zt[30:34] = np.nan  # no depth at all -> Sampson distance
    a, b = _views(uv_s, uv_t, zs, zt)
    cs, diag = match_observations(
        a, b, _bundle(reference_handeye_path), CFG.correspondence, CFG.depth, "track"
    )
    assert diag.model == "rigid"
    assert cs.inlier[40:48].all() and cs.inlier[30:34].all()
    assert [cs.rejection_reason[k] for k in bad[:4]] == ["REPROJECTION"] * 4
    assert np.isnan(cs.tgt_point_cam_m[40]).all() and np.isnan(cs.sigma_m[40])


def test_without_depth_falls_back_to_essential_matrix(reference_handeye_path: Path) -> None:
    _P, _Q, uv_s, uv_t, _zt, bad, _R, _t = _scene()
    nan = np.full(60, np.nan)
    a, b = _views(uv_s, uv_t, nan, nan)
    cs, diag = match_observations(
        a, b, _bundle(reference_handeye_path), CFG.correspondence, CFG.depth, "track"
    )
    assert diag.model == "essential"
    # an outlier can land near its epipolar line by chance, so only most must be rejected
    assert cs.inlier[: 60 - 12].mean() > 0.95 and cs.inlier[bad].mean() <= 0.25
    few_a, few_b = _views(uv_s[:5], uv_t[:5], nan[:5], nan[:5])
    few, d2 = match_observations(
        few_a, few_b, _bundle(reference_handeye_path), CFG.correspondence, CFG.depth, "track"
    )
    assert d2.model == "none" and set(few.rejection_reason) == {"NO_MODEL"}


def test_descriptor_candidates_ratio_and_mutual(reference_handeye_path: Path) -> None:
    P, _Q, uv_s, uv_t, zt, _bad, _R, _t = _scene(outliers=0)
    rng = np.random.default_rng(9)
    desc = rng.integers(0, 256, (60, 32), dtype=np.uint8)
    a, b = _views(uv_s, uv_t, P[:, 2], zt, ids_t=np.arange(100, 160, dtype=np.int64), desc=desc)
    cfg = replace(CFG.correspondence, max_hamming=64)
    cs, diag = match_observations(a, b, _bundle(reference_handeye_path), cfg, CFG.depth, "orb")
    assert diag.candidates == 60 and cs.inlier.all()
    assert np.all(cs.track_ids == -1)  # different track ids on the two sides
