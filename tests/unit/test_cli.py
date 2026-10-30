"""CLI wiring: --version, inspect-session, verify-session, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvsr.cli import build_parser, main
from mvsr.record.recorder import RecordRequest
from unit.test_io import make_session


def test_record_cli_defaults_are_one_p0_take() -> None:
    args = build_parser().parse_args(["record", "--object", "cup", "--gt", "constant_T_base_obj"])
    req = RecordRequest.from_cli(args)
    assert req.cameras == ("wrist", "head")
    assert req.scenario == ("R-STATIC",)
    assert req.duration_s is None
    assert req.gt_type == "constant_T_base_obj"
    assert req.preview is True
    assert build_parser().parse_args(["record", "--object", "cup", "--no-preview"]).preview is False


def test_tuning_cli_parses_and_params_status_agrees() -> None:
    assert build_parser().parse_args(["params-status"]).command == "params-status"
    assert build_parser().parse_args(["derive-params", "--stage", "sensors"]).stage == "sensors"
    assert main(["params-status"]) == 0


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("mvsr 0.1.0")


def test_inspect_and_verify(
    tmp_path: Path, reference_handeye_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d, _ = make_session(tmp_path, reference_handeye_path)
    assert (
        main(["verify-session", d.name, "--data-root", str(tmp_path), "--json", "--p0-gate"]) == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["images_checked"]
    assert out["stats"]["p0_gate"]["wrist_frames_written"] == 10
    assert main(["inspect-session", str(d), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["cameras"]["wrist"]["frames_seen"] == 12
    assert inspected["measured"]["cameras"]["wrist"]["frames_written"] == 10
    assert inspected["measured"]["duration_s"] is not None
    (d / "events.jsonl").write_text("")
    assert main(["verify-session", str(d), "--fast"]) == 1
    assert main(["verify-session", "20200101T000000_none_0000", "--data-root", str(tmp_path)]) == 2
