import numpy as np
import pytest

from object_reconstruction.fusion.keyframe_selector import KeyframeSelector


def pose(tx=0.0, ty=0.0, tz=0.0, yaw_deg=0.0):
    T = np.eye(4)
    yaw = np.radians(yaw_deg)
    T[:3, :3] = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    T[:3, 3] = [tx, ty, tz]
    return T


GOOD_AREA = 5000
GOOD_RATIO = 0.9


def make_selector():
    return KeyframeSelector(
        translation_m=0.02, rotation_deg=5.0,
        min_mask_area_px=1000, min_valid_depth_ratio=0.7,
    )


def test_first_valid_frame_accepted():
    selector = make_selector()
    assert selector.should_add(pose(), GOOD_AREA, GOOD_RATIO) is True


def test_small_motion_rejected_until_threshold():
    selector = make_selector()
    assert selector.should_add(pose(), GOOD_AREA, GOOD_RATIO)
    # 1 cm and 2 deg: below both thresholds.
    assert selector.should_add(pose(tx=0.01, yaw_deg=2.0), GOOD_AREA, GOOD_RATIO) is False
    # 3 cm translation: accepted, reference pose updates.
    assert selector.should_add(pose(tx=0.03), GOOD_AREA, GOOD_RATIO) is True
    # Another 1 cm relative to the NEW reference: rejected again.
    assert selector.should_add(pose(tx=0.04), GOOD_AREA, GOOD_RATIO) is False


def test_rotation_alone_triggers_keyframe():
    selector = make_selector()
    assert selector.should_add(pose(), GOOD_AREA, GOOD_RATIO)
    assert selector.should_add(pose(yaw_deg=6.0), GOOD_AREA, GOOD_RATIO) is True


def test_quality_gates_reject_bad_masks():
    selector = make_selector()
    # Poor quality frames never become keyframes nor move the reference.
    assert selector.should_add(pose(), mask_area_px=500, valid_depth_ratio=0.9) is False
    assert selector.should_add(pose(), mask_area_px=5000, valid_depth_ratio=0.5) is False
    # The next good frame is still treated as the first keyframe.
    assert selector.should_add(pose(), GOOD_AREA, GOOD_RATIO) is True


def test_pose_delta_values_and_clamp():
    t, r = KeyframeSelector.pose_delta(pose(), pose(tx=0.03, yaw_deg=10.0))
    assert t == pytest.approx(0.03)
    assert r == pytest.approx(10.0, abs=1e-9)
    # Identity delta must survive float noise in acos (trace ~ 3 + eps).
    T_noisy = np.eye(4)
    T_noisy[0, 0] = 1.0 + 1e-15
    t, r = KeyframeSelector.pose_delta(np.eye(4), T_noisy)
    assert t == 0.0
    assert r == pytest.approx(0.0)


def test_reset_clears_reference():
    selector = make_selector()
    assert selector.should_add(pose(), GOOD_AREA, GOOD_RATIO)
    selector.reset()
    # After reset even an identical pose is a fresh first keyframe.
    assert selector.should_add(pose(), GOOD_AREA, GOOD_RATIO) is True


def test_from_config():
    selector = KeyframeSelector.from_config(
        {"keyframe": {"translation_m": 0.05, "rotation_deg": 10.0,
                      "min_mask_area_px": 2000, "min_valid_depth_ratio": 0.8}}
    )
    assert selector.translation_m == 0.05
    assert selector.rotation_deg == 10.0
    assert selector.min_mask_area_px == 2000
    assert selector.min_valid_depth_ratio == 0.8
