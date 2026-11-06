"""Human-readable P4 keyframe-memory report."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _stat(value: dict[str, Any] | None, *keys: str) -> str:
    if not value:
        return "n/a"
    return " ".join(f"{key}={value[key]:g}" for key in keys)


def format_report(
    reports: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    run_dir: Path | None,
) -> str:
    out: list[str] = []
    for report in reports:
        keyframes, retrieval = report["keyframes"], report["retrieval"]
        reuse = report["revisit_reuse_ratio"]
        reuse_text = "n/a" if reuse is None else f"{reuse:.4f}"
        out.extend(
            [
                f"session {report['session_id']} scenario={(report.get('scenario') or {}).get('tags')}",
                f"  frames={report['frames']['processed']} keyframes={keyframes['count']} decisions={keyframes['decisions']}",
                f"  insert gaps {_stat(keyframes['insert_gap_frames'], 'median', 'p95', 'max')} frames; "
                f"revisit reuse={reuse_text}",
                f"  best-anchor hit {_stat(retrieval['best_anchor_hit_rate'], 'median', 'p95', 'min')}; "
                f"local size {_stat(retrieval['local_size'], 'median', 'max')}",
                f"  reproducibility: {report['reproducibility']['frames']} frames identical={report['reproducibility']['identical']}; "
                f"persistence reload={report.get('persistence', {}).get('reload_identical')}",
                f"  runtime={report['runtime']['wall_s']} s ({report['runtime']['fps']} fps)",
            ]
        )
        spread = report.get("static_spread")
        if spread:
            out.append(
                f"  tracking with P4 memory: translation {_stat(spread['translation_mm'], 'median', 'p95')} mm; "
                f"rotation {_stat(spread['rotation_deg'], 'median', 'p95')} deg"
            )
    if decision:
        out.append(f"P4 decision: {decision['decision']}")
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
