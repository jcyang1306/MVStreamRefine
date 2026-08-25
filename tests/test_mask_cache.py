import numpy as np

from object_reconstruction.segmentation.mask_cache import MaskCache


def test_save_and_load_roundtrip(tmp_path):
    cache = MaskCache(tmp_path / "masks")
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True

    path = cache.save(7, mask)
    assert path == tmp_path / "masks" / "cam" / "000007.png"

    loaded = cache.get(7)
    assert loaded.dtype == bool
    assert np.array_equal(loaded, mask)


def test_missing_mask_returns_none(tmp_path):
    cache = MaskCache(tmp_path / "masks")
    assert cache.get(0) is None
    assert cache.frames() == []


def test_frames_listing(tmp_path):
    cache = MaskCache(tmp_path / "masks")
    mask = np.ones((4, 4), dtype=bool)
    for idx in (5, 0, 12):
        cache.save(idx, mask)
    assert cache.frames() == [0, 5, 12]


def test_legacy_cam2_read_fallback(tmp_path):
    cache = MaskCache(tmp_path / "masks")
    legacy = cache.legacy_mask_path(9)
    legacy.parent.mkdir(parents=True)
    from PIL import Image

    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8)).save(legacy)
    assert cache.get(9).all()
    assert cache.frames() == [9]

    cache.save(10, np.ones((3, 4), dtype=bool))
    assert cache.frames() == [9, 10]


def test_metadata_roundtrip(tmp_path):
    cache = MaskCache(tmp_path / "masks")
    records = [
        {"frame": 0, "mask_area": 24500, "valid_depth_ratio": 0.91, "iou_prev": None},
        {"frame": 1, "mask_area": 24010, "valid_depth_ratio": 0.90, "iou_prev": 0.97},
    ]
    cache.write_metadata(records)
    assert cache.read_metadata() == records
