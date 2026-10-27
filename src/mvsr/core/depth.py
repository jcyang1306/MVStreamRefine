"""The single raw-depth -> meters conversion (ADR-0003). No other code scales depth."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from mvsr.core.errors import ContractValidationError


def depth_raw_to_m(
    depth_raw_u16: npt.NDArray[np.uint16], depth_units_m: float
) -> npt.NDArray[np.float32]:
    """`depth_m = raw * depth_units_m` computed in float64, stored float32; `0` stays `0.0` (invalid)."""
    raw = np.asarray(depth_raw_u16)
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ContractValidationError(f"depth must be uint16 HxW, got {raw.dtype} {raw.shape}")
    if not (
        isinstance(depth_units_m, float) and math.isfinite(depth_units_m) and depth_units_m > 0
    ):
        raise ContractValidationError(
            f"depth_units_m must be a positive float, got {depth_units_m!r}"
        )
    out: npt.NDArray[np.float32] = (raw.astype(np.float64) * depth_units_m).astype(np.float32)
    return out
