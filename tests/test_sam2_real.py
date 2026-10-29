"""Real SAM 2.1 tiny on real cup frames (needs CUDA + the pinned checkpoint; else SKIP).

The prompt box comes from the first frame of the registered R-STATIC take; this checks the
adapter against the real model, not mask accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvsr.io.registry import load_registry, resolve_fixture
from mvsr.observation.config import load_observation_config
from mvsr.observation.masks import MaskCache
from mvsr.observation.segmentation import (
    MaskPrompt,
    SegmentationError,
    segment_session,
    verify_checkpoint,
)

torch = pytest.importorskip("torch")
pytest.importorskip("sam2_inference")

CUP_SESSION = "20261028T080558_CW20175_9fb3"
CUP_BOX = (585.0, 31.0, 756.0, 269.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_stream_tracker_segments_the_cup(tmp_path_factory: pytest.TempPathFactory) -> None:
    cfg = load_observation_config()
    try:
        verify_checkpoint(cfg.sam2)
    except SegmentationError as exc:
        pytest.skip(str(exc))
    entry = next((e for e in load_registry() if e["session_id"] == CUP_SESSION), None)
    if entry is None:
        pytest.skip("cup session not registered")
    try:
        d = resolve_fixture(entry)
    except FileNotFoundError:
        pytest.skip("cup session missing")
    out = tmp_path_factory.mktemp("sam")
    meta = segment_session(
        d, MaskPrompt(box_xyxy=CUP_BOX, source="test"), cfg, limit=8, cache_root=out
    )
    cache = MaskCache(out, meta["mask_revision"])
    areas = [cache.records[f].area_px for f in cache.frame_ids()]
    assert len(areas) == 8 and min(areas) > 5_000
    first = cache.mask(cache.frame_ids()[0])
    ys, xs = np.nonzero(first)
    assert 560 < xs.mean() < 780 and 20 < ys.mean() < 290
    assert all(cache.records[f].status.value == "OK" for f in cache.frame_ids())
