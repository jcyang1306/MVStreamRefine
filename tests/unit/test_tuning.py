from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mvsr.tuning import derive
from mvsr.tuning.registry import reminder_lines, status


def test_fit_quadratic_sigma_recovers_b() -> None:
    z = np.linspace(0.3, 0.45, 50)
    fit = derive.fit_quadratic_sigma(z, 0.02 * z**2)
    assert fit["b_per_m"] == pytest.approx(0.02)
    assert fit["rms_residual_m"] < 1e-12
    assert fit["free_fit"]["a_m"] == pytest.approx(0.0, abs=1e-12)
    assert fit["z_range_m"] == pytest.approx([0.3, 0.45])


def test_max_interp_gap_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    iv = np.full(10_000, 10.0)
    iv[:20] = 25.4  # 0.2 % of intervals -> above p99.9
    monkeypatch.setattr(derive, "robot_intervals_ms", lambda dirs: iv)
    out = derive.max_interp_gap(["x"])
    assert out["value_ns"] == 26_000_000
    assert out["beyond_value"] == 0

    monkeypatch.setattr(derive, "robot_intervals_ms", lambda dirs: np.full(100, 10.2))
    assert derive.max_interp_gap(["x"])["value_ns"] == 21_000_000  # 2 x median floor


def _entry(**kw: Any) -> dict[str, Any]:
    e = {
        "key": "sensors.sync.max_interp_gap_ns",
        "value": 0,
        "derived_on": "2026-10-30",
        "basis": ["a"],
        "rule": "r",
        "recheck_when": {"scenarios": ["R-STATIC"], "min_new": 1},
    }
    e.update(kw)
    return e


def test_status_due_on_new_matching_session() -> None:
    sessions = [
        {"session_id": "a", "scenario": ["R-STATIC"]},
        {"session_id": "b", "scenario": ["R-CALIB"]},
    ]
    (row,) = status([_entry()], sessions)
    assert not row["due"] and row["new_sessions"] == []
    sessions.append({"session_id": "c", "scenario": ["R-STATIC"]})
    (row,) = status([_entry()], sessions)
    assert row["due"] and row["new_sessions"] == ["c"]
    assert row["config_mismatch"]  # value 0 differs from configs/sensors.yaml
    lines = reminder_lines([row])
    assert "sensors.sync.max_interp_gap_ns" in lines[0] and "c" in lines[1]


def test_status_min_new() -> None:
    sessions = [{"session_id": "c", "scenario": ["R-STATIC"]}]
    (row,) = status([_entry(recheck_when={"scenarios": ["R-STATIC"], "min_new": 2})], sessions)
    assert not row["due"]
    assert reminder_lines([row]) == []
