import pytest

from sam2_inference._common import MODEL_CONFIGS, build_video_predictor


@pytest.mark.parametrize("model_type", MODEL_CONFIGS)
def test_video_builder_model_mapping(tmp_path, model_type):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    calls = []

    def builder(config, path, **kwargs):
        calls.append((config, path, kwargs))
        return object()

    result = build_video_predictor(
        str(checkpoint), model_type, "cpu", builder=builder
    )
    assert result is not None
    assert calls == [
        (
            MODEL_CONFIGS[model_type],
            str(checkpoint),
            {"device": "cpu", "vos_optimized": False},
        )
    ]


def test_builder_rejects_bad_model_and_missing_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="model_type"):
        build_video_predictor(str(checkpoint), "medium", "cpu")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        build_video_predictor(str(tmp_path / "missing.pt"), "tiny", "cpu")
