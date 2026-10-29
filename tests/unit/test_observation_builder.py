"""ObservationBuilder and the benchmark on a hand-built textured session (code paths only)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from mvsr.core.types import ObjectObservation
from mvsr.observation.benchmark import i1_decision, observation_digest, run_benchmark, write_run
from mvsr.observation.builder import ObservationBuilder, ObservationError
from mvsr.observation.config import load_observation_config
from mvsr.observation.masks import MaskCache, MaskCacheWriter, MaskStatus, mask_revision_id
from mvsr.sensors.replay import ReplaySource
from unit.test_replay import make_session

H, W = 120, 160


def _images(i: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(21)
    base = rng.integers(0, 256, (H // 4 + 4, W // 4 + 4), dtype=np.uint8)
    tex = cv2.GaussianBlur(
        cv2.resize(base, ((W + 16), (H + 16)), interpolation=cv2.INTER_NEAREST), (3, 3), 0
    )
    g = np.ascontiguousarray(tex[8 : 8 + H, 8 - (i % 3) : 8 - (i % 3) + W])
    depth = np.full((H, W), 4000, np.uint16)
    depth[:, :10] = 0
    return np.stack([g, g, g], axis=2), depth


def _mask() -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[20:100, 30:130] = True
    return m


@pytest.fixture
def session_with_masks(tmp_path: Path, reference_handeye_path: Path) -> tuple[Path, str]:
    d = make_session(
        tmp_path,
        reference_handeye_path,
        n=10,
        gt_type="constant_T_base_obj",
        size=(W, H),
        images=_images,
    )
    rev = mask_revision_id("test", {"case": "builder"})
    w = MaskCacheWriter(d, "wrist", rev, {"params": {"prompt": {"frame_id": 0}}})
    for fid in ReplaySource(d).frame_ids:
        empty = fid == 6
        m = np.zeros((H, W), bool) if empty else _mask()
        w.add(fid, m, MaskStatus.EMPTY if empty else MaskStatus.OK, None if empty else 0.95)
    w.finalize()
    return d, rev


def test_builder_produces_contract_observations(session_with_masks: tuple[Path, str]) -> None:
    d, rev = session_with_masks
    cfg = load_observation_config()
    src = ReplaySource(d)
    b = ObservationBuilder(MaskCache(d, rev), src.bundle, cfg)
    obs = [b.process(src.read(f)) for f in src.frame_ids]
    assert all(isinstance(o, ObjectObservation) for o in obs)
    first = obs[0]
    assert first.mask_revision == rev and first.descriptor_type == "orb" and first.mask_prob is None
    assert first.keypoints_uv.shape[0] > 10 and first.descriptors.shape[1] == 32
    assert first.points_cam_m.shape[0] == first.geometry_mask.sum() > 0
    np.testing.assert_allclose(first.points_cam_m[:, 2], 0.4, atol=1e-6)
    assert np.isfinite(first.keypoint_depth_m).any()
    assert not (first.geometry_mask & ~first.mask_binary).any()
    empty = next(o for o in obs if o.header.frame_id == 6)
    assert empty.keypoints_uv.shape[0] == 0 and empty.points_cam_m.shape[0] == 0
    assert empty.quality["mask_status"] == "EMPTY" and empty.quality["tracks_occluded"] > 0
    later = next(o for o in obs if o.header.frame_id == 2)
    assert set(later.track_ids.tolist()) & set(first.track_ids.tolist())
    with pytest.raises(ObservationError, match="increasing"):
        b.process(src.read(src.frame_ids[0]))


def test_builder_rejects_missing_masks_and_is_deterministic(
    session_with_masks: tuple[Path, str],
) -> None:
    d, rev = session_with_masks
    cfg = load_observation_config()
    src = ReplaySource(d)
    partial = mask_revision_id("test", {"case": "partial"})
    w = MaskCacheWriter(d, "wrist", partial, {})
    w.add(0, _mask(), MaskStatus.OK, 0.9)
    w.finalize()
    b = ObservationBuilder(MaskCache(d, partial), src.bundle, cfg)
    with pytest.raises(ObservationError, match="no mask"):
        b.process(src.read(1))
    run = [
        observation_digest(
            ObservationBuilder(MaskCache(d, rev), src.bundle, cfg).process(src.read(0))
        )
        for _ in range(2)
    ]
    assert run[0] == run[1]


def test_benchmark_and_gate_on_hand_built_session(
    session_with_masks: tuple[Path, str], tmp_path: Path
) -> None:
    d, rev = session_with_masks
    cfg = load_observation_config()
    cfg = replace(
        cfg,
        benchmark=replace(
            cfg.benchmark, frame_gaps=[1, 2], pair_stride=1, reproducibility_frames=4
        ),
    )
    rep = run_benchmark(d, rev, cfg)
    assert rep["frames"]["observed"] == 9 and rep["frames"]["mask_coverage"] == 1.0
    assert rep["frames"]["mask_status"] == {"EMPTY": 1, "OK": 8}
    assert rep["reproducibility"] == {
        "frames": 4,
        "identical": True,
        "first_digest": rep["reproducibility"]["first_digest"],
    }
    assert rep["reference_available"] and "gap_1" in rep["pairs"] and rep["tracks"]["count"] > 0
    decision = i1_decision([rep], [["R-STATIC"]], cfg)
    assert decision["decision"] == "BLOCKED" and decision["failed_hard"] == ["r_hard_session"]
    ok = i1_decision([rep], [["R-STATIC"], ["R-HARD"]], cfg)
    assert "r_hard_session" not in ok["failed_hard"]
    run = write_run([rep], decision, cfg, tmp_path)
    assert (run / "observation_report.json").is_file() and run.name.split("_")[1] == "CW20175"
