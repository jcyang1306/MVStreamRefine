"""Optional OpenCV preview window with live stats (display only; never affects recording).

Keys: SPACE pause/resume, A annotation event, Q / ESC stop.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from mvsr.core.errors import DeviceError

WINDOW = "mvsr record"
_KEYS = {ord(" "): "space", ord("q"): "q", 27: "esc", ord("a"): "a"}


class Preview:
    def __init__(self, max_width: int = 960) -> None:
        if not os.environ.get("DISPLAY"):
            raise DeviceError(
                "--preview needs an X display (DISPLAY unset; run `xhost +si:localuser:$USER`)"
            )
        self.max_width = int(max_width)
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        except cv2.error as exc:
            raise DeviceError(f"cannot open preview window: {exc}") from exc

    def show(
        self,
        rgb: np.ndarray | None,
        lines: list[str],
        recording: bool,
        views: dict[str, np.ndarray | None] | None = None,
    ) -> str | None:
        canvas: np.ndarray
        mosaic = _mosaic_bgr(views) if views else None
        if mosaic is None and rgb is not None:
            mosaic = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if mosaic is None:
            canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            scale = min(1.0, self.max_width / mosaic.shape[1])
            canvas = cv2.resize(mosaic, None, fx=scale, fy=scale) if scale < 1.0 else mosaic
        color = (0, 0, 255) if recording else (0, 200, 255)
        y = 26
        for line in [*lines, "SPACE pause/resume  A annotate  Q stop"]:
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y += 24
        cv2.imshow(WINDOW, canvas)
        return _KEYS.get(cv2.waitKey(1) & 0xFF)

    def close(self) -> None:
        try:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass


def _mosaic_bgr(views: dict[str, np.ndarray | None]) -> np.ndarray | None:
    tiles: list[np.ndarray] = []
    for name, rgb in views.items():
        if rgb is None:
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, name, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
        cv2.putText(bgr, name, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        tiles.append(bgr)
    if not tiles:
        return None
    h = min(im.shape[0] for im in tiles)
    aligned = [
        cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h)) if im.shape[0] != h else im
        for im in tiles
    ]
    return np.hstack(aligned)
