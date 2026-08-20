import numpy as np
import pytest

from object_reconstruction.geometry.rgbd import erode_mask, preprocess_object_rgbd

CONFIG = {
    "mask": {"erosion_px": 1},
    "depth": {"scale": 1000.0, "min_m": 0.15, "max_m": 1.2},
}


def make_frame(h=10, w=12):
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    depth = np.full((h, w), 500, dtype=np.uint16)  # 0.5 m everywhere
    mask = np.zeros((h, w), dtype=bool)
    mask[2:8, 3:9] = True  # 6 x 6 block
    return rgb, depth, mask


def test_erosion_shrinks_boundary():
    _, _, mask = make_frame()
    eroded = erode_mask(mask, 1)
    assert eroded.sum() == 4 * 4  # 6x6 block loses its 1-px border
    assert eroded[3:7, 4:8].all()
    assert not eroded[2, :].any()
    # erosion_px=0 is a no-op.
    assert np.array_equal(erode_mask(mask, 0), mask)


def test_depth_zeroed_outside_eroded_mask():
    rgb, depth, mask = make_frame()
    result = preprocess_object_rgbd(rgb, depth, mask, CONFIG)
    assert result.depth.dtype == np.uint16
    assert (result.depth[result.mask] == 500).all()
    assert (result.depth[~result.mask] == 0).all()
    assert result.mask_area_px == 16
    assert result.valid_depth_px == 16
    assert result.valid_depth_ratio == 1.0
    # RGB is passed through untouched.
    assert result.rgb is rgb


def test_depth_range_and_invalid_filtering():
    rgb, depth, mask = make_frame()
    depth[4, 5] = 0      # invalid
    depth[4, 6] = 100    # 0.1 m, below min
    depth[5, 5] = 3000   # 3 m, above max
    result = preprocess_object_rgbd(rgb, depth, mask, CONFIG)
    assert result.depth[4, 5] == 0
    assert result.depth[4, 6] == 0
    assert result.depth[5, 5] == 0
    assert result.mask_area_px == 16
    assert result.valid_depth_px == 13
    assert result.valid_depth_ratio == pytest.approx(13 / 16)


def test_empty_mask_gives_zero_ratio():
    rgb, depth, _ = make_frame()
    empty = np.zeros(depth.shape, dtype=bool)
    result = preprocess_object_rgbd(rgb, depth, empty, CONFIG)
    assert result.mask_area_px == 0
    assert result.valid_depth_ratio == 0.0
    assert not result.depth.any()


def test_misaligned_shapes_rejected():
    rgb, depth, mask = make_frame()
    with pytest.raises(ValueError, match="pixel aligned"):
        preprocess_object_rgbd(rgb, depth[:-1], mask[:-1], CONFIG)


def test_null_depth_scale_forbidden():
    rgb, depth, mask = make_frame()
    config = {"mask": {"erosion_px": 0}, "depth": {"scale": None}}
    with pytest.raises(ValueError, match="depth.scale"):
        preprocess_object_rgbd(rgb, depth, mask, config)