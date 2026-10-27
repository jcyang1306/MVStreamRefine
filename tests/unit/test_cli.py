"""CLI wiring: --version, inspect-session, verify-session, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvsr.cli import main
from unit.test_io import make_session


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("mvsr 0.1.0")


def test_inspect_and_verify(
    tmp_path: Path, reference_handeye_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d, _ = make_session(tmp_path, reference_handeye_path)
    assert main(["verify-session", d.name, "--data-root", str(tmp_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["images_checked"]
    assert main(["inspect-session", str(d), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cameras"]["wrist"]["frames_seen"] == 12
    (d / "events.jsonl").write_text("")
    assert main(["verify-session", str(d), "--fast"]) == 1
    assert main(["verify-session", "20200101T000000_none_0000", "--data-root", str(tmp_path)]) == 2
