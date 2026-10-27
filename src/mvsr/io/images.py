"""Lossless PNG encode/decode for session images (ADR-0003).

Color PNGs are standard 8-bit RGB files (OpenCV's BGR order is converted at the boundary);
depth PNGs are 16-bit single-channel files holding the raw device value.
"""

from __future__ import annotations

import hashlib
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.core.errors import SessionFormatError

#: zlib level 1 = fastest lossless compression (ADR-0003 "fast compression level").
DEFAULT_PNG_COMPRESSION = 1


def raw_sha256(arr: np.ndarray) -> str:
    """sha256 of the C-contiguous array bytes (pre-encoding content hash)."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def encode_png_rgb(rgb: npt.NDArray[np.uint8], level: int = DEFAULT_PNG_COMPRESSION) -> bytes:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"color must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(level)])
    if not ok:
        raise SessionFormatError("PNG encode failed (color)")
    return buf.tobytes()


def encode_png_u16(depth: npt.NDArray[np.uint16], level: int = DEFAULT_PNG_COMPRESSION) -> bytes:
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"depth must be uint16 HxW, got {depth.dtype} {depth.shape}")
    ok, buf = cv2.imencode(
        ".png", np.ascontiguousarray(depth), [cv2.IMWRITE_PNG_COMPRESSION, int(level)]
    )
    if not ok:
        raise SessionFormatError("PNG encode failed (depth)")
    return buf.tobytes()


def decode_png_rgb(data: bytes) -> npt.NDArray[np.uint8]:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        shape = None if img is None else (img.dtype, img.shape)
        raise SessionFormatError(f"color PNG is not 8-bit RGB: {shape}")
    return cast(npt.NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def decode_png_u16(data: bytes) -> npt.NDArray[np.uint16]:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint16 or img.ndim != 2:
        shape = None if img is None else (img.dtype, img.shape)
        raise SessionFormatError(f"depth PNG is not 16-bit single channel: {shape}")
    return cast(npt.NDArray[np.uint16], img)
