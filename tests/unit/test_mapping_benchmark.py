"""P5 benchmark writes provisional artifacts and blocks unauthorized commit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from mvsr.mapping.benchmark import i3_decision, run_benchmark, write_run
from mvsr.mapping.config import load_mapping_config
from unit.test_keyframe_manager import _run
from unit.test_tracking_session import _make


def test_mapping_benchmark_is_provisional_and_blocks_i3(
    tmp_path: Path, reference_handeye_path: Path
) -> None:
    session, revision = _make(tmp_path, reference_handeye_path)
    manager, _ = _run(session, revision)
    memory = manager.persist(tmp_path / "keyframe-memory")
    config = load_mapping_config()
    config = replace(
        config,
        benchmark=replace(config.benchmark, heldout_stride=2),
        gate=replace(config.gate, min_geometry_points=10),
    )
    result = run_benchmark(session, memory, config, manager.cfg)
    report = result.report
    assert report["geometry"]["points"] > 0
    assert report["geometry"]["deterministic"]
    assert report["commit_mode"] == "PROVISIONAL_DIAGNOSTIC"
    decision = i3_decision([report], config)
    assert decision["decision"] == "BLOCKED"
    assert "r_both_session" in decision["failed_hard"]
    body: dict[str, Any] = {}
    run = write_run([result], body, config, tmp_path / "runs")
    assert (run / "mapping_report.json").is_file()
    assert report["artifacts"]["mode"] == "PROVISIONAL_DIAGNOSTIC"
    assert not (run / "mapping" / report["session_id"] / "committed").exists()
