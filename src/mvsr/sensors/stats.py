"""Summary statistics used by P1 reports (deterministic, JSON-ready)."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np


def summary(
    values: Sequence[float] | np.ndarray, scale: float = 1.0, digits: int = 4
) -> dict[str, float | int] | None:
    """`n, min, median, p95, max, mean, rmse` of `values * scale`; None when empty."""
    v = np.asarray(values, dtype=np.float64) * scale
    if v.size == 0:
        return None
    return {
        "n": int(v.size),
        "min": round(float(v.min()), digits),
        "median": round(float(np.median(v)), digits),
        "p95": round(float(np.percentile(v, 95)), digits),
        "max": round(float(v.max()), digits),
        "mean": round(float(v.mean()), digits),
        "rmse": round(float(np.sqrt(np.mean(v * v))), digits),
    }


def histogram(values: Sequence[float] | np.ndarray, edges: Sequence[float]) -> dict[str, int]:
    """Counts per `[edges[i], edges[i+1])` plus an overflow bin `>=edges[-1]`."""
    v = np.asarray(values, dtype=np.float64)
    out: dict[str, int] = {}
    for lo, hi in pairwise(edges):
        out[f"{lo:g}-{hi:g}"] = int(((v >= lo) & (v < hi)).sum())
    out[f">={edges[-1]:g}"] = int((v >= edges[-1]).sum())
    return out
