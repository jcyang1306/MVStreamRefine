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

    def show(self, rgb: np.ndarray | None, lines: list[str], recording: bool) -> str | None:
        canvas: np.ndarray
        if rgb is None:
            canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            scale = min(1.0, self.max_width / rgb.shape[1])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            canvas = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1.0 else bgr.copy()
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
