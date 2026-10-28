"""Preview mosaic layout (no X11 window)."""

from __future__ import annotations

import numpy as np

from mvsr.record.preview import _mosaic_bgr


def test_mosaic_side_by_side_wrist_then_head() -> None:
    wrist = np.zeros((40, 80, 3), dtype=np.uint8)
    head = np.full((40, 80, 3), 255, dtype=np.uint8)
    mosaic = _mosaic_bgr({"wrist": wrist, "head": head})
    assert mosaic is not None and mosaic.shape == (40, 160, 3)
    assert mosaic[20, 10].sum() < mosaic[20, 90].sum()


def test_mosaic_skips_missing_and_empty() -> None:
    assert _mosaic_bgr({"wrist": None, "head": None}) is None
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    mosaic = _mosaic_bgr({"wrist": None, "head": rgb})
    assert mosaic is not None and mosaic.shape[1] == 12
