"""Live rig smoke tests (F6). Run on CW20175 only:

    docker compose run --rm rig pytest -q tests/hardware --run-hardware

They record 10 s sessions into a temporary data root and require `verify-session` to pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mvsr.io.inspect import inspect_session
from mvsr.io.verify import verify_session
from mvsr.record.config import load_record_config
from mvsr.record.recorder import RecordRequest, run_recorder

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("cameras", [("wrist",), ("wrist", "head")], ids=["wrist", "wrist+head"])
def test_record_10s_verifies(tmp_path: Path, cameras: tuple[str, ...]) -> None:
    req = RecordRequest(
        cameras=cameras,
        scenario=("SMOKE",),
        object_name="hardware-smoke",
        duration_s=10.0,
        notes="pytest hardware smoke",
        data_root=tmp_path,
    )
    res = run_recorder(load_record_config(), req, install_signal_handlers=False)
    assert res.status == "COMPLETE", res.errors
    rep = verify_session(res.session_dir)
    assert rep.ok, rep.errors
    report = inspect_session(res.session_dir)
    for cid in cameras:
        cam = report["cameras"][cid]
        print(cid, cam["written_ratio"], cam["arrival_fps"], cam["timestamp_field_coverage"])
        assert cam["frames_written"] > 0
    print("robot", report["robot"]["rate_hz"], report["robot"]["interval"])
    assert report["robot"]["samples"] > 0
