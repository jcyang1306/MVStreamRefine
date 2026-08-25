from types import SimpleNamespace

import numpy as np
import pytest

from tools import precompute_masks


class FakeCV2:
    IMREAD_COLOR = 1
    WINDOW_NORMAL = 0
    error = RuntimeError

    def __init__(self, roi=(10, 20, 30, 40)):
        self.roi = roi
        self.destroyed = False

    def imread(self, _path, _mode):
        return np.zeros((100, 200, 3), dtype=np.uint8)

    def namedWindow(self, _name, _mode):
        pass

    def selectROI(self, _name, _image, showCrosshair, fromCenter):
        assert showCrosshair is True
        assert fromCenter is False
        return self.roi

    def destroyAllWindows(self):
        self.destroyed = True


def test_interactive_roi_converts_xywh_to_xyxy(monkeypatch, tmp_path):
    fake_cv2 = FakeCV2()
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    box = precompute_masks.select_interactive_box(tmp_path / "first.jpg", "cam")

    assert box == (10.0, 20.0, 40.0, 60.0)
    assert fake_cv2.destroyed is True


def test_interactive_roi_rejects_cancel(monkeypatch, tmp_path):
    fake_cv2 = FakeCV2(roi=(0, 0, 0, 0))
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    with pytest.raises(RuntimeError, match="cancelled or empty"):
        precompute_masks.select_interactive_box(tmp_path / "first.jpg", "cam")


def test_resolve_box_interactively(monkeypatch, tmp_path):
    source = SimpleNamespace(
        root=tmp_path,
        indices=[0],
        cam_prefix="wrist",
    )
    calls = []

    def fake_select(path, name):
        calls.append((path, name))
        return (1.0, 2.0, 3.0, 4.0)

    monkeypatch.setattr(precompute_masks, "select_interactive_box", fake_select)
    box = precompute_masks.resolve_box(source, None)

    assert box == (1.0, 2.0, 3.0, 4.0)
    assert calls == [
        (tmp_path / "frame-000000_wrist_color.jpg", "cam (wrist)")
    ]


def test_resolve_box_uses_cli_without_interaction(monkeypatch, tmp_path):
    source = SimpleNamespace(root=tmp_path, indices=[0], cam_prefix="wrist")
    monkeypatch.setattr(
        precompute_masks,
        "select_interactive_box",
        lambda *_: pytest.fail("interactive selection should not run"),
    )
    assert precompute_masks.resolve_box(source, [10, 20, 30, 40]) == (
        10, 20, 30, 40
    )
