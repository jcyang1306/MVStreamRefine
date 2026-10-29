"""ORB extraction and KLT tracks on a seeded texture shifted by known amounts (P2 O3/O4)."""

from __future__ import annotations

import cv2
import numpy as np

from mvsr.observation.config import load_observation_config
from mvsr.observation.features import OrbExtractor, make_extractor
from mvsr.observation.tracks import TrackManager, TrackStatus

H, W = 240, 320


def _texture() -> np.ndarray:
    rng = np.random.default_rng(11)
    base = rng.integers(0, 256, (H // 4, W // 4), dtype=np.uint8)
    img = cv2.resize(base, (W + 40, H + 40), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _view(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.ascontiguousarray(img[20 - dy : 20 - dy + H, 20 - dx : 20 - dx + W])


def _safe() -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[40:200, 60:260] = True
    return m


def test_orb_detects_inside_mask_and_describes_given_points() -> None:
    cfg = load_observation_config().features
    ext = make_extractor(cfg)
    assert isinstance(ext, OrbExtractor) and ext.descriptor_type == "orb"
    img = _view(_texture(), 0, 0)
    uv, desc = ext.detect(img, _safe())
    assert uv.shape[0] > 50 and desc.shape == (uv.shape[0], 32) and desc.dtype == np.uint8
    assert np.all(_safe()[np.rint(uv[:, 1]).astype(int), np.rint(uv[:, 0]).astype(int)])
    pts = np.array([[100.0, 100.0], [1.0, 1.0], [150.5, 120.25]], np.float32)
    idx, d2 = ext.describe(img, pts)
    assert idx.tolist() == [0, 2] and d2.shape == (2, 32)
    empty_uv, empty_desc = ext.detect(img, np.zeros((H, W), bool))
    assert empty_uv.shape == (0, 2) and empty_desc.shape == (0, 32)


def test_tracks_follow_a_known_shift_with_stable_ids() -> None:
    cfg = load_observation_config()
    tm = TrackManager(cfg.tracks, make_extractor(cfg.features))
    tex = _texture()
    f0 = tm.update(0, _view(tex, 0, 0), _safe())
    assert f0.uv.shape[0] > 50 and all(s is TrackStatus.NEW for s in f0.status)
    f1 = tm.update(1, _view(tex, 3, -2), _safe())
    tracked = [k for k, s in enumerate(f1.status) if s is TrackStatus.TRACKED]
    assert len(tracked) > 0.8 * f0.uv.shape[0]
    pos0 = {int(t): f0.uv[k] for k, t in enumerate(f0.track_ids)}
    shifts = np.array([f1.uv[k] - pos0[int(f1.track_ids[k])] for k in tracked])
    np.testing.assert_allclose(np.median(shifts, axis=0), [3.0, -2.0], atol=0.1)
    assert np.all(f1.confidence[tracked] > 0.0)
    assert len(set(f1.track_ids.tolist())) == f1.track_ids.shape[0]


def test_occlusion_then_retirement_and_ids_never_reused() -> None:
    cfg = load_observation_config()
    tm = TrackManager(cfg.tracks, make_extractor(cfg.features))
    tex = _texture()
    f0 = tm.update(0, _view(tex, 0, 0), _safe())
    first_ids = set(f0.track_ids.tolist())
    none = np.zeros((H, W), bool)
    occl = tm.update(1, _view(tex, 0, 0), none)
    assert occl.uv.shape[0] == 0 and set(occl.occluded_ids) == first_ids
    for k in range(2, 2 + cfg.tracks.max_occluded_frames):
        last = tm.update(k, _view(tex, 0, 0), none)
    assert set(last.retired_ids) == first_ids and not last.occluded_ids
    again = tm.update(20, _view(tex, 0, 0), _safe())
    assert again.uv.shape[0] > 0 and not (set(again.track_ids.tolist()) & first_ids)
    assert all(s is TrackStatus.NEW for s in again.status)
