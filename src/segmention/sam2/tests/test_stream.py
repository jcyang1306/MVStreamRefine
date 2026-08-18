import numpy as np
import pytest
import torch

from sam2_inference import StreamTracker


class FakeStreamPredictor:
    image_size = 4
    num_maskmem = 2

    def _get_image_feature(self, state, frame_idx, batch_size):
        state["cached_features"][frame_idx] = object()

    def _register(self, state, obj_id):
        if obj_id in state["obj_id_to_idx"]:
            return
        index = len(state["obj_id_to_idx"])
        state["obj_id_to_idx"][obj_id] = index
        state["obj_idx_to_id"][index] = obj_id
        state["obj_ids"] = list(state["obj_id_to_idx"])
        state["point_inputs_per_obj"][index] = {}
        state["mask_inputs_per_obj"][index] = {}
        state["output_dict_per_obj"][index] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        state["temp_output_dict_per_obj"][index] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        state["frames_tracked_per_obj"][index] = {}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **_kwargs):
        self._register(inference_state, obj_id)
        return frame_idx, inference_state["obj_ids"], torch.ones(
            len(inference_state["obj_ids"]), 1, 6, 8
        )

    def add_new_mask(self, state, frame_idx, obj_id, _mask):
        self._register(state, obj_id)
        return frame_idx, state["obj_ids"], torch.ones(len(state["obj_ids"]), 1, 6, 8)

    def propagate_in_video_preflight(self, _state):
        pass

    def _run_single_frame_inference(self, **_kwargs):
        logits = torch.ones(1, 1, 4, 4)
        return {"pred_masks": logits}, logits

    def _get_orig_video_res_output(self, state, logits):
        resized = torch.nn.functional.interpolate(
            logits, size=(state["video_height"], state["video_width"])
        )
        return logits, resized

    def reset_state(self, state):
        state["obj_ids"].clear()


def test_stream_two_frames_and_multiple_objects():
    tracker = StreamTracker(predictor=FakeStreamPredictor(), device="cpu")
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    tracker.add_prompt(frame, object_id=3, box=[0, 0, 4, 4])
    tracker.add_prompt(object_id=9, points=[[2, 2]], labels=[1])
    result = tracker.track(frame)
    assert result.frame_index == 1
    assert result.object_ids == (3, 9)
    assert result.masks.shape == (2, 6, 8)
    assert result.masks.dtype == np.bool_
    tracker.close()
    assert tracker.state is None


def test_stream_requires_prompt_and_fixed_shape():
    tracker = StreamTracker(predictor=FakeStreamPredictor(), device="cpu")
    with pytest.raises(RuntimeError, match="add_prompt"):
        tracker.track(np.zeros((6, 8, 3), dtype=np.uint8))
    tracker.add_prompt(
        np.zeros((6, 8, 3), dtype=np.uint8), box=[0, 0, 4, 4]
    )
    with pytest.raises(ValueError, match="resolution"):
        tracker.track(np.zeros((7, 8, 3), dtype=np.uint8))


def test_stream_memory_validation():
    with pytest.raises(ValueError, match=">= 4"):
        StreamTracker(
            predictor=FakeStreamPredictor(), device="cpu", memory_frames=3
        )
