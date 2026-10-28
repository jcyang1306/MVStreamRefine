"""Parser for the reference hand-eye file `handeye_tf.txt` (ADR-0007, COORDINATES_AND_TIMING.md).

File format: labeled blocks `<label>:` followed by 16 numbers (a row-major 4x4 matrix, any
brackets/commas/whitespace). Block semantics (owner-confirmed):
- `wrist_cam2` = `T_tcp_cam`  (wrist D405 color optical frame in the RM65 tool frame)
- `base_cam1`  = `T_base_head` (fixed D435 color optical frame in the robot base)
The reference is `configs/handeye/handeye_tf_20261028.txt`, taken from
`calibration_result.json` (`cam2toeef`, `cam1tobase`; session `real_train_007`, method PARK,
quality `exploratory`). The 2026-10-12 file stays in the tree for the sessions that copied it.
A different checksum raises `ChecksumMismatchError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mvsr.core.errors import ChecksumMismatchError, SessionFormatError
from mvsr.core.se3 import PoseSE3
from mvsr.io.atomic import sha256_bytes

REFERENCE_HANDEYE_SHA256 = "2b60289b9b68c62b08d40eddc0a06232249c152d12ec1687639e6c2d974d0ef3"
REFERENCE_HANDEYE_DATE = "2026-10-28"
WRIST_LABEL = "wrist_cam2"
HEAD_LABEL = "base_cam1"

_LABEL = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*$", re.MULTILINE)
_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_handeye_text(text: str) -> dict[str, np.ndarray]:
    """Return `{label: float64[4,4]}`; every block must hold exactly 16 numbers."""
    matches = list(_LABEL.finditer(text))
    if not matches:
        raise SessionFormatError("hand-eye file has no '<label>:' blocks")
    if text[: matches[0].start()].strip():
        raise SessionFormatError("hand-eye file has content before the first label")
    out: dict[str, np.ndarray] = {}
    for i, m in enumerate(matches):
        label = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end]
        leftover = _FLOAT.sub("", body)
        if re.sub(r"[\s\[\],]", "", leftover):
            raise SessionFormatError(f"block '{label}' has unexpected characters: {leftover!r}")
        values = [float(v) for v in _FLOAT.findall(body)]
        if len(values) != 16:
            raise SessionFormatError(f"block '{label}' has {len(values)} numbers, expected 16")
        if label in out:
            raise SessionFormatError(f"duplicate block '{label}'")
        out[label] = np.array(values, dtype=np.float64).reshape(4, 4)
    return out


@dataclass(frozen=True)
class HandEye:
    """Extrinsics from one hand-eye file plus its provenance."""

    T_tcp_cam: PoseSE3
    T_base_head: PoseSE3 | None
    sha256: str
    source_path: str
    raw_text: str


def load_handeye(
    path: str | Path, expected_sha256: str | None = REFERENCE_HANDEYE_SHA256
) -> HandEye:
    """Load and validate; `expected_sha256=None` disables the checksum pin (tests only)."""
    p = Path(path)
    data = p.read_bytes()
    digest = sha256_bytes(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ChecksumMismatchError(
            f"{p}: sha256 {digest} != reference {expected_sha256} (ADR-0007)"
        )
    text = data.decode("utf-8")
    blocks = parse_handeye_text(text)
    if WRIST_LABEL not in blocks:
        raise SessionFormatError(f"{p}: missing block '{WRIST_LABEL}' (T_tcp_cam)")
    head = blocks.get(HEAD_LABEL)
    return HandEye(
        T_tcp_cam=PoseSE3(blocks[WRIST_LABEL], "tcp", "cam"),
        T_base_head=None if head is None else PoseSE3(head, "base", "head"),
        sha256=digest,
        source_path=str(p),
        raw_text=text,
    )
