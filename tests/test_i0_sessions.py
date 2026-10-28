"""I0 checks on registered real sessions (fixtures/registry.yaml `used_by: I0`, ADR-0002).

A missing or checksum-mismatched fixture is an explicit SKIP here; the gate itself
(`mvsr i0-gate`) is BLOCKED in that case.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from mvsr.core.errors import ChecksumMismatchError
from mvsr.core.types import deep_equal
from mvsr.io.registry import load_registry, resolve_fixture
from mvsr.sensors.board import BoardSpec, rotation_angle_deg
from mvsr.sensors.calib_check import observe_stream
from mvsr.sensors.clock import MID_EXPOSURE_METHODS
from mvsr.sensors.config import load_sensors_config
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.report import i0_decision, session_report

ENTRIES = [e for e in load_registry() if "I0" in e["used_by"]]
CALIB = [e for e in ENTRIES if "R-CALIB" in e["scenario"]]


def _fixture(entry: dict[str, Any]) -> Any:
    try:
        return resolve_fixture(entry)
    except FileNotFoundError:
        pytest.skip(f"fixture {entry['session_id']} missing")
    except ChecksumMismatchError as exc:
        pytest.skip(f"fixture {entry['session_id']} checksum mismatch: {exc}")


@pytest.mark.skipif(not ENTRIES, reason="no I0 sessions registered")
@pytest.mark.parametrize("entry", ENTRIES, ids=[e["session_id"] for e in ENTRIES])
def test_replay_is_deterministic_with_mid_exposure_clock(entry: dict[str, Any]) -> None:
    d = _fixture(entry)
    ids = ReplaySource(d).frame_ids[:20]
    a = list(ReplaySource(d, frame_ids=ids))
    b = list(ReplaySource(d, frame_ids=ids))
    assert all(deep_equal(x, y) for x, y in zip(a, b, strict=True))
    src = ReplaySource(d)
    assert set(src.clock.report["methods"]) <= set(MID_EXPOSURE_METHODS)
    ts = [src.clock.timestamp(f)[0] for f in src.frame_ids]
    assert all(t1 > t0 for t0, t1 in itertools.pairwise(ts))


@pytest.mark.skipif(not CALIB, reason="no R-CALIB session registered for I0")
@pytest.mark.parametrize("entry", CALIB, ids=[e["session_id"] for e in CALIB])
def test_board_orientation_is_consistent(entry: dict[str, Any]) -> None:
    d = _fixture(entry)
    cfg = load_sensors_config()
    cfg = replace(cfg, board=replace(cfg.board, stride=40))
    obs, _, _ = observe_stream(ReplaySource(d, config=cfg), BoardSpec(11, 8, 0.015), cfg)
    poses = [o.T_base_cam_prior @ o.T_cam_marker for o in obs if o.T_base_cam_prior is not None]
    assert len(poses) >= 10
    assert max(rotation_angle_deg(poses[0].R, p.R) for p in poses) < 5.0
    assert max(float(np.linalg.norm(p.t - poses[0].t)) for p in poses) < 0.02


@pytest.mark.skipif(not CALIB, reason="no R-CALIB session registered for I0")
@pytest.mark.parametrize("entry", CALIB, ids=[e["session_id"] for e in CALIB])
def test_i0_gate_is_not_blocked(entry: dict[str, Any]) -> None:
    d = _fixture(entry)
    cfg = load_sensors_config()
    cfg = replace(
        cfg, board=replace(cfg.board, stride=10), gate=replace(cfg.gate, min_board_frames=50)
    )
    rep, bundles = session_report(d, cfg)
    decision = i0_decision([rep], cfg)
    assert decision["decision"] != "BLOCKED", decision["failed_hard"]
    assert "wrist" in bundles and "P1 S2" in bundles["wrist"].notes
