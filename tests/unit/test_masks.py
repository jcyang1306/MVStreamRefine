"""Mask PNG IO and the offline mask cache (P2 O1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mvsr.core.errors import ChecksumMismatchError, SessionFormatError
from mvsr.io import images
from mvsr.observation.masks import (
    MaskCache,
    MaskCacheWriter,
    MaskStatus,
    list_revisions,
    mask_bbox,
    mask_revision_id,
)


def _mask(h: int = 12, w: int = 16) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[3:8, 4:11] = True
    return m


def test_mask_png_round_trip_and_validation() -> None:
    m = _mask()
    np.testing.assert_array_equal(images.decode_png_mask(images.encode_png_mask(m)), m)
    with pytest.raises(ValueError):
        images.encode_png_mask(m.astype(np.uint8))
    gray = images.encode_png_u16(np.full((4, 4), 3, np.uint16))
    with pytest.raises(SessionFormatError):
        images.decode_png_mask(gray)


def test_revision_id_is_deterministic_and_parameter_sensitive() -> None:
    a = mask_revision_id("sam2.1-tiny", {"box": [1, 2, 3, 4], "n": 5})
    assert a == mask_revision_id("sam2.1-tiny", {"n": 5, "box": [1, 2, 3, 4]})
    assert a != mask_revision_id("sam2.1-tiny", {"box": [1, 2, 3, 5], "n": 5})
    assert a.startswith("sam2.1-tiny@") and len(a.split("@")[1]) == 12
    with pytest.raises(ValueError):
        mask_revision_id("bad name", {})
    assert mask_bbox(_mask()) == (4, 3, 11, 8) and mask_bbox(np.zeros((2, 2), bool)) is None


def test_cache_write_read_and_immutability(tmp_path: Path) -> None:
    rev = mask_revision_id("test", {"x": 1})
    w = MaskCacheWriter(tmp_path, "wrist", rev, {"params": {"x": 1}})
    w.add(0, _mask(), MaskStatus.OK, 0.93)
    w.add(2, np.zeros((12, 16), bool), MaskStatus.EMPTY, None)
    with pytest.raises(SessionFormatError):
        w.add(2, _mask(), MaskStatus.OK, 0.9)
    meta = w.finalize()
    assert meta["by_status"] == {"EMPTY": 1, "OK": 1} and meta["frames"] == 2
    assert list_revisions(tmp_path) == [rev]
    cache = MaskCache(tmp_path, rev)
    assert cache.frame_ids() == [0, 2] and 2 in cache and 1 not in cache
    np.testing.assert_array_equal(cache.mask(0), _mask())
    assert cache.records[0].area_px == 35 and cache.records[0].bbox_xyxy == (4, 3, 11, 8)
    assert not cache.mask(0).flags.writeable
    with pytest.raises(SessionFormatError):
        cache.mask(1)
    with pytest.raises(FileExistsError):
        MaskCacheWriter(tmp_path, "wrist", rev, {})


def test_cache_detects_tampering_and_ignores_partial_runs(tmp_path: Path) -> None:
    rev = mask_revision_id("test", {"x": 2})
    w = MaskCacheWriter(tmp_path, "wrist", rev, {})
    w.add(0, _mask(), MaskStatus.OK, 0.9)
    w.finalize()
    png = MaskCache(tmp_path, rev).dir / "000000.png"
    png.write_bytes(images.encode_png_mask(~_mask()))
    with pytest.raises(ChecksumMismatchError):
        MaskCache(tmp_path, rev).mask(0)
    partial = mask_revision_id("test", {"x": 3})
    pw = MaskCacheWriter(tmp_path, "wrist", partial, {})
    pw.add(0, _mask(), MaskStatus.OK, 0.9)
    assert partial not in list_revisions(tmp_path)
    with pytest.raises(SessionFormatError, match="incomplete"):
        MaskCache(tmp_path, partial)
    MaskCacheWriter(tmp_path, "wrist", partial, {}).finalize()  # interrupted dir is replaced
    assert partial in list_revisions(tmp_path)
