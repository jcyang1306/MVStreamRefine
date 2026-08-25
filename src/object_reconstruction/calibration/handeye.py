"""Parser for labeled 4x4 matrices in data/handeye/handeye_tf.txt.

The single-camera system consumes the legacy physical label ``wrist_cam2`` as
``T_tcp_cam``. Additional legacy matrices may remain in the file and are
ignored by the caller. Matrix direction is confirmed via handeye_convention.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def load_handeye_matrices(path: str | Path) -> dict[str, np.ndarray]:
    text = Path(path).read_text()
    labels = re.findall(r"(\w+)\s*:", text)
    if not labels:
        raise ValueError(f"{path}: no labeled matrices found")
    matrices: dict[str, np.ndarray] = {}
    blocks = re.split(r"\w+\s*:", text)[1:]
    for label, block in zip(labels, blocks):
        values = [float(v) for v in _FLOAT.findall(block)]
        if len(values) != 16:
            raise ValueError(f"{path}: block '{label}' has {len(values)} values, expected 16")
        matrices[label] = np.array(values, dtype=np.float64).reshape(4, 4)
    return matrices
