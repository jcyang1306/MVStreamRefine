"""SAM 2.1 adapter control flow with a tracker double (no GPU; ADR-0002: not accuracy evidence)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mvsr.core.errors import ChecksumMismatchError
from mvsr.observation.config import load_observation_config
from mvsr.observation.masks import MaskCache, MaskStatus
from mvsr.observation.segmentation import (
    MaskPrompt,
    SegmentationError,
    classify,
    segment_session,
    verify_checkpoint,
)
from unit.test_replay import make_session


class BoxTracker:
    """Returns the prompt box as the mask; frame k >= 4 loses the object."""

    def __init__(self) -> None:
        self.box: list[float] | None = None
        self.calls: list[str] = []
        self.k = 0
        self.was_reset = False

    def _result(self, rgb: np.ndarray) -> Any:
        h, w = rgb.shape[:2]
        logits = np.full((1, h, w), -4.0, dtype=np.float32)
        if self.k < 4 and self.box is not None:
            x1, y1, x2, y2 = (int(v) for v in self.box)
            logits[0, y1:y2, x1:x2] = 3.0
        self.k += 1
        return SimpleNamespace(masks=logits > 0, logits=logits)

    def add_prompt(
        self, rgb: np.ndarray, *, object_id: int, box: Any, points: Any, labels: Any
    ) -> Any:
        assert object_id == 1 and rgb.dtype == np.uint8
        self.box = box
        self.calls.append("add_prompt")
        return self._result(rgb)

    def track(self, rgb: np.ndarray) -> Any:
        self.calls.append("track")
        return self._result(rgb)

    def reset(self) -> None:
        self.was_reset = True


@pytest.fixture
def session(tmp_path: Path, reference_handeye_path: Path) -> Path:
    return make_session(tmp_path, reference_handeye_path, n=8)


def test_segment_session_writes_cache(session: Path) -> None:
    cfg = load_observation_config()
    cfg = replace(cfg, sam2=replace(cfg.sam2, min_mask_area_px=4))
    tracker = BoxTracker()
    meta = segment_session(
        session, MaskPrompt(box_xyxy=(2, 2, 10, 8)), cfg, tracker_factory=lambda s: tracker
    )
    assert (
        tracker.calls[0] == "add_prompt"
        and set(tracker.calls[1:]) == {"track"}
        and tracker.was_reset
    )
    cache = MaskCache(session, meta["mask_revision"])
    assert cache.frame_ids() == [0, 1, 2, 4, 5, 6, 7]  # frame 3 was dropped at record time
    status = {f: cache.records[f].status for f in cache.frame_ids()}
    assert [status[f] for f in (0, 1, 2, 4)] == [MaskStatus.OK] * 4
    assert [status[f] for f in (5, 6, 7)] == [MaskStatus.EMPTY] * 3
    assert (
        cache.records[0].area_px == 48
        and abs(cache.records[0].score - 1 / (1 + np.exp(-3.0))) < 1e-6
    )
    assert meta["params"]["prompt"]["frame_id"] == 0 and meta["params"]["n_frames"] == 7
    with pytest.raises(FileExistsError):
        segment_session(
            session, MaskPrompt(box_xyxy=(2, 2, 10, 8)), cfg, tracker_factory=lambda s: BoxTracker()
        )
    later = segment_session(
        session,
        MaskPrompt(frame_id=4, box_xyxy=(2, 2, 10, 8)),
        cfg,
        tracker_factory=lambda s: BoxTracker(),
    )
    assert later["mask_revision"] != meta["mask_revision"]
    assert MaskCache(session, later["mask_revision"]).frame_ids() == [4, 5, 6, 7]


def test_prompt_and_frame_validation(session: Path) -> None:
    cfg = load_observation_config()
    with pytest.raises(SegmentationError):
        MaskPrompt()
    with pytest.raises(SegmentationError):
        MaskPrompt(box_xyxy=(5, 5, 2, 9))
    with pytest.raises(SegmentationError):
        MaskPrompt(points=((1.0, 2.0),), labels=(2,))
    with pytest.raises(SegmentationError, match="not an emitted frame"):
        segment_session(
            session,
            MaskPrompt(frame_id=3, box_xyxy=(1, 1, 4, 4)),
            cfg,
            tracker_factory=lambda s: BoxTracker(),
        )


def test_classify_and_checkpoint_checks(tmp_path: Path) -> None:
    s = load_observation_config().sam2
    m = np.zeros((40, 40), bool)
    assert classify(m, None, s) is MaskStatus.EMPTY
    m[:, :] = True
    assert (
        classify(m, 0.95, s) is MaskStatus.OK and classify(m, 0.3, s) is MaskStatus.LOW_CONFIDENCE
    )
    with pytest.raises(SegmentationError, match="not found"):
        verify_checkpoint(replace(s, checkpoint=str(tmp_path / "none.pt")))
    fake = tmp_path / "ck.pt"
    fake.write_bytes(b"not a checkpoint")
    with pytest.raises(ChecksumMismatchError):
        verify_checkpoint(replace(s, checkpoint=str(fake)))
