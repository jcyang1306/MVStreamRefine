"""Human-readable P3 tracking report."""

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
        frames = report["frames"]
        keyframes = report["keyframes"]
        spread = report.get("static_spread")
        out.extend(
            [
                f"session {report['session_id']} scenario={(report.get('scenario') or {}).get('tags')}",
                f"  poses={frames['processed']} valid_ratio={frames['valid_ratio']:.4f} usable_ratio={frames['usable_ratio']:.4f} status={frames['status']}",
                f"  states={frames['track_state']} reasons={frames['reason_codes']}",
                f"  keyframes inserted={keyframes['inserted']} retained={keyframes['retained']} "
                f"anchor age {_stat(keyframes['anchor_age_frames'], 'median', 'p95', 'max')} frames",
                f"  distance to anchor {_stat(keyframes['distance_to_anchor_translation_mm'], 'median', 'p95')} mm, "
                f"{_stat(keyframes['distance_to_anchor_rotation_deg'], 'median', 'p95')} deg",
            ]
        )
        if spread:
            out.append(
                f"  R-STATIC T_base_obj spread: translation {_stat(spread['translation_mm'], 'median', 'p95', 'max')} mm; "
                f"rotation {_stat(spread['rotation_deg'], 'median', 'p95', 'max')} deg"
            )
        ablations = report["ablations"]
        for name in ("frame_to_frame_spread", "icp_only_spread"):
            value = ablations.get(name)
            if value:
                out.append(
                    f"  {name}: translation {_stat(value['translation_mm'], 'median', 'p95')} mm; "
                    f"rotation {_stat(value['rotation_deg'], 'median', 'p95')} deg"
                )
        repro = report["reproducibility"]
        out.append(
            f"  reproducibility: {repro['frames']} frames identical={repro['identical']}; "
            f"runtime={report['runtime']['wall_s']} s ({report['runtime']['fps']} fps)"
        )
    if decision:
        out.append(f"I2 decision: {decision['decision']}")
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
