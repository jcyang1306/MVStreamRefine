"""Human-readable P5 mapping report."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _residual(value: dict[str, Any] | None) -> str:
    if not value:
        return "n/a"
    stats = value["distance_mm"]
    return f"median={stats['median']:g} p95={stats['p95']:g} mm coverage={value['coverage']:.4f}"


def format_report(
    reports: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    run_dir: Path | None,
) -> str:
    out: list[str] = []
    for report in reports:
        geometry = report["geometry"]
        spread = report.get("tracking_spread")
        out.extend(
            [
                f"session {report['session_id']} scenario={(report.get('scenario') or {}).get('tags')}",
                f"  keyframes train={len(report['keyframes']['train'])} heldout={len(report['keyframes']['heldout'])} "
                f"tracking_authorized={report['tracking_authorized_for_commit']}",
                f"  provisional geometry points={geometry['points']} vertices={geometry['vertices']} triangles={geometry['triangles']} "
                f"deterministic={geometry['deterministic']}",
                f"  heldout {_residual(report.get('heldout'))}",
                f"  robot-prior static baseline {_residual(report.get('robot_prior_static_baseline'))}",
                f"  wrong-inverse baseline {_residual(report.get('wrong_inverse_baseline'))} "
                f"ratio={report.get('wrong_inverse_error_ratio')}",
            ]
        )
        if spread:
            out.append(
                f"  selected-pose spread p95={spread['translation_mm']['p95']} mm / "
                f"{spread['rotation_deg']['p95']} deg"
            )
        out.append(f"  artifact mode={report['commit_mode']}")
    if decision:
        out.append(f"I3 decision: {decision['decision']}")
        for check in decision["checks"]:
            flag = "ok " if check["ok"] else ("FAIL" if check["hard"] else "miss")
            hard = ", hard" if check["hard"] else ""
            out.append(
                f"  {flag} {check['name']} = {check['value']} (target {check['target']}{hard})"
            )
        out.append("  scope limits: " + "; ".join(decision["scope_limits"]))
    if run_dir is not None:
        out.append(f"run written to {run_dir}")
    return "\n".join(out)
