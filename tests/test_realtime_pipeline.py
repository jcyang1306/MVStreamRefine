"""RealtimePipeline state-machine tests with fake engine and segmenter."""

import numpy as np
import pytest

from object_reconstruction.data.models import CameraFrame, CameraIntrinsics, FramePacket
from object_reconstruction.pipeline.realtime_pipeline import RealtimePipeline, State
from object_reconstruction.pipeline.reconstruction_engine import EngineResult

HEIGHT, WIDTH = 48, 64

CONFIG = {
    "realtime": {
        "tracking": {"min_mask_area_px": 100, "lost_after_frames": 3},
    }
}


class FakeEngine:
    _created = 0

    def __init__(self):
        FakeEngine._created += 1
        self.id = FakeEngine._created
        self.keyframes = 0
        self.processed = []

    def process(self, packet, mask):
        self.processed.append(packet.index)
        self.keyframes += 1
        return EngineResult(integrated=True, rgbd=None, keyframe_id=self.keyframes)


class FakeSegmenter:
    def __init__(self, track_areas=None):
        # Queue of mask areas returned by successive track() calls.
        self.track_areas = list(track_areas or [])
        self.resets = 0
        self.started_with = []
        self.active = False

    def _mask(self, area):
        mask = np.zeros(HEIGHT * WIDTH, dtype=bool)
        mask[:area] = True
        return mask.reshape(HEIGHT, WIDTH)

    def start_object(self, rgb, box):
        self.started_with.append(box)
        self.active = True
        return self._mask(500)

    def track(self, rgb):
        area = self.track_areas.pop(0) if self.track_areas else 500
        return self._mask(area)

    def reset(self):
        self.resets += 1
        self.active = False


def make_packet(index=0):
    return FramePacket(
        index=index,
        timestamp=None,
        cam=CameraFrame(
            rgb=np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8),
            depth=np.full((HEIGHT, WIDTH), 500, dtype=np.uint16),
            intrinsics=CameraIntrinsics(WIDTH, HEIGHT, 50.0, 50.0, 32.0, 24.0),
        ),
        T_world_cam=np.eye(4),
    )


def make_pipeline(segmenter=None):
    return RealtimePipeline(FakeEngine, segmenter or FakeSegmenter(), CONFIG)


def test_preview_does_not_track_or_integrate():
    pipeline = make_pipeline()
    output = pipeline.process(make_packet())
    assert output.state == State.PREVIEW
    assert output.mask is None
    assert pipeline.engine.processed == []


def test_roi_confirm_run_cycle():
    segmenter = FakeSegmenter()
    pipeline = make_pipeline(segmenter)

    mask = pipeline.set_roi(np.zeros((HEIGHT, WIDTH, 3), np.uint8), (1, 2, 30, 40))
    assert pipeline.state == State.MASK_CONFIRM
    assert segmenter.started_with == [(1, 2, 30, 40)]
    assert mask.sum() == 500

    # MASK_CONFIRM tracks (so the object is followed) but never integrates.
    output = pipeline.process(make_packet(0))
    assert output.state == State.MASK_CONFIRM
    assert output.mask is not None and output.engine_result is None
    assert pipeline.engine.processed == []

    assert pipeline.handle_key("r") == "mask confirmed; integrating"
    output = pipeline.process(make_packet(1))
    assert output.state == State.RUNNING
    assert output.engine_result.integrated
    assert pipeline.engine.processed == [1]


def test_pause_keeps_tracking_without_integration():
    pipeline = make_pipeline()
    pipeline.set_roi(np.zeros((HEIGHT, WIDTH, 3), np.uint8), (0, 0, 8, 8))
    pipeline.handle_key("r")
    pipeline.process(make_packet(0))

    assert "paused" in pipeline.handle_key("p")
    output = pipeline.process(make_packet(1))
    assert output.state == State.PAUSED
    assert output.mask is not None  # still tracked
    assert pipeline.engine.processed == [0]

    assert pipeline.handle_key("p") == "resumed"
    pipeline.process(make_packet(2))
    assert pipeline.engine.processed == [0, 2]


def test_lost_after_consecutive_small_masks_requires_new_roi():
    segmenter = FakeSegmenter(track_areas=[500, 10, 10, 10])
    pipeline = make_pipeline(segmenter)
    pipeline.set_roi(np.zeros((HEIGHT, WIDTH, 3), np.uint8), (0, 0, 8, 8))
    pipeline.handle_key("r")

    pipeline.process(make_packet(0))  # area 500: fine
    pipeline.process(make_packet(1))  # 10 (streak 1)
    pipeline.process(make_packet(2))  # 10 (streak 2)
    output = pipeline.process(make_packet(3))  # 10 (streak 3) -> LOST
    assert output.state == State.LOST
    assert "lost" in output.message
    assert segmenter.resets == 1  # tracking stopped

    # LOST frames are passthrough until a new ROI arrives; model is kept.
    engine_before = pipeline.engine
    assert pipeline.process(make_packet(4)).mask is None
    pipeline.set_roi(np.zeros((HEIGHT, WIDTH, 3), np.uint8), (0, 0, 8, 8))
    assert pipeline.state == State.MASK_CONFIRM
    assert pipeline.engine is engine_before


def test_recovered_mask_resets_lost_streak():
    segmenter = FakeSegmenter(track_areas=[10, 10, 500, 10, 10])
    pipeline = make_pipeline(segmenter)
    pipeline.set_roi(np.zeros((HEIGHT, WIDTH, 3), np.uint8), (0, 0, 8, 8))
    pipeline.handle_key("r")
    for index in range(5):
        output = pipeline.process(make_packet(index))
    assert output.state == State.RUNNING  # streak was broken by the 500 frame


def test_clear_keeps_model_and_new_resets_engine():
    segmenter = FakeSegmenter()
    pipeline = make_pipeline(segmenter)
    pipeline.set_roi(np.zeros((HEIGHT, WIDTH, 3), np.uint8), (0, 0, 8, 8))
    pipeline.handle_key("r")
    pipeline.process(make_packet(0))
    engine = pipeline.engine
    assert engine.keyframes == 1

    assert "model kept" in pipeline.handle_key("c")
    assert pipeline.state == State.PREVIEW
    assert pipeline.engine is engine  # TSDF untouched
    assert segmenter.resets == 1

    assert "new model" in pipeline.handle_key("n")
    assert pipeline.engine is not engine
    assert pipeline.engine.keyframes == 0


def test_unmapped_keys_are_ignored():
    pipeline = make_pipeline()
    assert pipeline.handle_key("x") is None
    assert pipeline.handle_key("r") is None  # r does nothing in PREVIEW
    assert pipeline.state == State.PREVIEW
