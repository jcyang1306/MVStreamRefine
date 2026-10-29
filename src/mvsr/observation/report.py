"""Human-readable summary of an observation report (P2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _s(stats: dict[str, Any] | None, *keys: str) -> str:
    if not stats:
        return "n/a"
    return " ".join(f"{k}={stats[k]:g}" for k in keys)


def format_report(
    reports: list[dict[str, Any]], decision: dict[str, Any] | None, run_dir: Path | None
) -> str:
    out: list[str] = []
    for r in reports:
        sc = r.get("scenario") or {}
        fr, mk, dp, kp, tr = r["frames"], r["mask"], r["depth"], r["keypoints"], r["tracks"]
        out += [
            f"session {r['session_id']}  scenario={sc.get('tags')} object={sc.get('object')!r}  mask_revision={r['mask_revision']}",
            f"  frames observed={fr['observed']}/{fr['expected_from_prompt']} coverage={fr['mask_coverage']} mask_status={fr['mask_status']}",
            f"  mask area {_s(mk['area_px'], 'min', 'median', 'max')} px, score {_s(mk['score'], 'min', 'median')}",
            f"  depth valid {_s(dp['valid_depth_ratio'], 'median', 'min')}, geometry {_s(dp['geometry_ratio'], 'median')}, "
            f"silhouette depth-step offset {_s(dp['silhouette_offset_px_points'], 'median', 'p95')} px (frames {_s(dp['silhouette_offset_px_per_frame'], 'n')})",
            f"  keypoints/frame {_s(kp['per_frame'], 'median', 'min')}, with depth {_s(kp['with_depth_per_frame'], 'median')}, "
            f"tracked {_s(kp['tracked_per_frame'], 'median')}, new {_s(kp['new_per_frame'], 'median')}",
            f"  tracks {tr['count']}, life {_s(tr['life_frames'], 'median', 'p95', 'max')} frames, survival {tr['survival']}",
        ]
        for age, st in tr["reprojection_drift_px"].items():
            out.append(
                f"    drift age {age}: {_s(st, 'median', 'p95')} px (n={st['n'] if st else 0})"
            )
        if not r["reference_available"]:
            out.append("  pairs: no constant_T_base_obj reference in this session")
        for key, m in r["pairs"].items():
            out.append(
                f"  [{key}] mask agreement {_s(m.get('mask_agreement'), 'median')}; "
                f"track inliers {_s(m.get('track_inliers'), 'median')} precision {_s(m.get('track_precision'), 'median')} recall {_s(m.get('track_recall'), 'median')}; "
                f"orb inliers {_s(m.get('orb_inliers'), 'median')} precision {_s(m.get('orb_precision'), 'median')} recall {_s(m.get('orb_recall'), 'median')}"
            )
        for key, st in (r.get("inlier_residual_mm") or {}).items():
            out.append(
                f"  inlier 3D residual vs reference [{key}]: {_s(st, 'median', 'p95')} mm (n={st['n'] if st else 0})"
            )
        rp = r["reproducibility"]
        out.append(f"  reproducibility: {rp['frames']} frames identical={rp['identical']}")
    if decision:
        out.append(f"I1 decision: {decision['decision']}")
        for c in decision["checks"]:
            flag = "ok " if c["ok"] else ("FAIL" if c["hard"] else "miss")
            out.append(
                f"  {flag} {c['name']} = {c['value']} (target {c['target']}{', hard' if c['hard'] else ''})"
            )
        out.append("  scope limits: " + "; ".join(decision["scope_limits"]))
    if run_dir is not None:
        out.append(f"run written to {run_dir}")
    return "\n".join(out)
