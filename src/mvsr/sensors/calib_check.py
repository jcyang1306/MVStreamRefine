"""R-CALIB calibration verification (P1 S2) — the board part of the sensors report.

For a static board every accepted wrist frame gives `T_base_marker = T_base_cam_prior @
T_cam_marker`; its deviation from the session reference (median translation, chordal mean
rotation) measures hand-eye + robot + timing error together inside the recorded motion. The
same reference re-projected through each frame's prior gives the hand-eye reprojection
residual in pixels. A nearest-sample prior is evaluated alongside as a baseline.

Depth (per accepted frame, inside the inner-corner hull shrunk by `edge_margin_px`):
- bias: measured depth minus the PnP board plane along each pixel ray;
- planarity: RMS distance of the measured points to their own best-fit plane;
- scale: adjacent-corner distance from depth-backprojected corners vs. the nominal square.

The head camera (fixed) is checked with `T_base_head @ T_head_marker` against the wrist
reference. All numbers are evidence inside this session's envelope only (ADR-0004 §7).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from mvsr.core.se3 import PoseSE3
from mvsr.core.types import CAMERA_FRAME, CalibrationBundle, SensorFrame, SyncStatus
from mvsr.sensors.board import (
    BoardSpec,
    detect_corners,
    mean_pose,
    project_points,
    rotation_angle_deg,
    solve_board_pose,
    undistort_normalized,
)
from mvsr.sensors.config import DepthCheckConfig, SensorsConfig
from mvsr.sensors.replay import ReplaySource
from mvsr.sensors.stats import summary

_SQUARE_RE = re.compile(r"square_mm\s*=\s*([0-9]*\.?[0-9]+)")


def square_mm_from_notes(notes: str | None) -> float | None:
    m = _SQUARE_RE.search(notes or "")
    return float(m.group(1)) if m else None


@dataclass(frozen=True, eq=False)
class BoardObservation:
    frame_id: int
    timestamp_ns: int
    sync_status: str
    T_cam_marker: PoseSE3
    corners_px: npt.NDArray[np.float64]
    corners_undist_px: npt.NDArray[np.float64]
    reproj_rms_px: float
    T_base_cam_prior: PoseSE3 | None
    T_base_cam_nearest: PoseSE3 | None
    depth: dict[str, Any]


# --------------------------------------------------------------------------- depth


def _rays(uv: npt.NDArray[np.float64], bundle: CalibrationBundle) -> npt.NDArray[np.float64]:
    n = undistort_normalized(uv, bundle.K, bundle.distortion_model, bundle.distortion)
    return np.concatenate([n, np.ones((n.shape[0], 1))], axis=1)


def depth_check(
    frame: SensorFrame,
    bundle: CalibrationBundle,
    T_cam_marker: PoseSE3,
    corners_px: npt.NDArray[np.float64],
    spec: BoardSpec,
    cfg: DepthCheckConfig,
) -> dict[str, Any]:
    h, w = frame.depth_raw_u16.shape
    grid = corners_px.reshape(spec.inner_rows, spec.inner_cols, 2)
    quad = np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]])
    mask: npt.NDArray[Any] = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    if cfg.edge_margin_px:
        k = 2 * cfg.edge_margin_px + 1
        mask = cv2.erode(mask, np.ones((k, k), dtype=np.uint8))
    region = mask.astype(bool)
    area = int(region.sum())
    valid = region & frame.valid_depth_mask
    ratio = float(valid.sum()) / area if area else 0.0
    out: dict[str, Any] = {"valid_ratio": round(ratio, 4), "used": False}
    if area == 0 or ratio < cfg.min_valid_ratio:
        return out
    v, u = np.nonzero(valid)
    rays = _rays(np.stack([u, v], axis=1).astype(np.float64), bundle)
    z = frame.depth_m_f32[v, u].astype(np.float64)
    pts = rays * z[:, None]
    n_pl = T_cam_marker.R[:, 2]
    d_pl = float(n_pl @ T_cam_marker.t)
    z_plane = d_pl / (rays @ n_pl)
    bias = z - z_plane
    centered = pts - pts.mean(axis=0)
    normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
    plane_dist = np.abs(centered @ normal)
    plane_rms = float(np.sqrt(np.mean(plane_dist**2)))
    r = cfg.corner_window_px
    corner3d = np.full((spec.n_corners, 3), np.nan)
    corner_rays = _rays(corners_px, bundle)
    for i, (cu, cv_) in enumerate(np.round(corners_px).astype(int)):
        y0, y1, x0, x1 = max(cv_ - r, 0), min(cv_ + r + 1, h), max(cu - r, 0), min(cu + r + 1, w)
        win = frame.depth_m_f32[y0:y1, x0:x1]
        win = win[win > 0]
        if win.size:
            corner3d[i] = corner_rays[i] * float(np.median(win))
    c3 = corner3d.reshape(spec.inner_rows, spec.inner_cols, 3)
    steps = np.concatenate(
        [
            np.linalg.norm(np.diff(c3, axis=1), axis=2).ravel(),
            np.linalg.norm(np.diff(c3, axis=0), axis=2).ravel(),
        ]
    )
    steps = steps[np.isfinite(steps)]
    out.update(
        used=True,
        distance_m=round(float(np.median(z)), 5),
        bias_mm=round(float(np.median(bias)) * 1e3, 4),
        bias_mad_mm=round(float(np.median(np.abs(bias - np.median(bias)))) * 1e3, 4),
        plane_rms_mm=round(plane_rms * 1e3, 4),
        plane_abs_median_mm=round(float(np.median(plane_dist)) * 1e3, 4),
        plane_abs_p90_mm=round(float(np.percentile(plane_dist, 90)) * 1e3, 4),
        pitch_mm=round(float(np.median(steps)) * 1e3, 4) if steps.size else None,
        pitch_pairs=int(steps.size),
    )
    return out


# --------------------------------------------------------------------------- observations


def observe_frame(
    source: ReplaySource, frame_id: int, spec: BoardSpec, cfg: SensorsConfig
) -> tuple[int, BoardObservation | None, str | None]:
    frame = source.read(frame_id)
    gray = np.asarray(cv2.cvtColor(frame.color_rgb_u8, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    corners = detect_corners(gray, spec)
    if corners is None:
        return frame_id, None, "not_detected"
    b = source.bundle
    pose = solve_board_pose(
        corners, spec, b.K, b.distortion_model, b.distortion, CAMERA_FRAME[b.camera_id]
    )
    if pose is None:
        return frame_id, None, "pnp_failed"
    if pose.reproj_rms_px > cfg.board.max_reproj_rms_px:
        return frame_id, None, "reproj_rms_too_high"
    nearest: PoseSE3 | None = None
    if b.mount == "wrist":
        T_n, info = source.robot.nearest(
            frame.header.timestamp_ns, cfg.sync.max_interp_gap_ns, b.dt_robot_cam_ns
        )
        if T_n is not None and info.status is SyncStatus.INTERPOLATED:
            assert b.T_tcp_cam is not None
            nearest = T_n @ b.T_tcp_cam
    obs = BoardObservation(
        frame_id=frame_id,
        timestamp_ns=frame.header.timestamp_ns,
        sync_status=frame.pose_sync.status.value,
        T_cam_marker=pose.T_cam_marker,
        corners_px=pose.corners_px,
        corners_undist_px=pose.corners_undist_px,
        reproj_rms_px=pose.reproj_rms_px,
        T_base_cam_prior=frame.T_base_cam_prior,
        T_base_cam_nearest=nearest,
        depth=depth_check(frame, b, pose.T_cam_marker, corners, spec, cfg.depth),
    )
    return frame_id, obs, None


def observe_stream(
    source: ReplaySource, spec: BoardSpec, cfg: SensorsConfig
) -> tuple[list[BoardObservation], dict[str, int], int]:
    ids = source.frame_ids[:: cfg.board.stride]
    with ThreadPoolExecutor(max_workers=cfg.board.workers) as pool:
        results = list(pool.map(lambda f: observe_frame(source, f, spec, cfg), ids))
    rejected: dict[str, int] = {}
    obs = []
    for _, o, reason in sorted(results, key=lambda x: x[0]):
        if o is None:
            assert reason is not None
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            obs.append(o)
    return obs, dict(sorted(rejected.items())), len(ids)


# --------------------------------------------------------------------------- aggregation


def _spread(poses: list[PoseSE3], ref: PoseSE3) -> dict[str, Any]:
    dt = np.stack([p.t - ref.t for p in poses])
    return {
        "translation_mm": summary(np.linalg.norm(dt, axis=1), 1e3),
        "translation_abs_axis_median_mm": [
            round(float(v), 4) for v in np.median(np.abs(dt), axis=0) * 1e3
        ],
        "rotation_deg": summary([rotation_angle_deg(ref.R, p.R) for p in poses]),
    }


def _pose_json(p: PoseSE3) -> dict[str, Any]:
    return {"dst": p.dst, "src": p.src, "matrix": [[float(v) for v in row] for row in p.matrix]}


def _depth_summary(
    obs: list[BoardObservation], spec: BoardSpec, cfg: DepthCheckConfig
) -> dict[str, Any]:
    used = [o.depth for o in obs if o.depth.get("used")]
    if not used:
        return {"frames_used": 0, "frames_checked": len(obs)}
    dist = np.array([d["distance_m"] for d in used])
    bias = np.array([d["bias_mm"] for d in used])
    bins: dict[str, Any] = {}
    idx = np.floor(dist / cfg.bin_m).astype(int)
    for k in sorted(set(idx.tolist())):
        sel = idx == k
        bins[f"{k * cfg.bin_m:.2f}-{(k + 1) * cfg.bin_m:.2f}m"] = {
            "frames": int(sel.sum()),
            "bias_mm_median": round(float(np.median(bias[sel])), 4),
        }
    pitch = np.array([d["pitch_mm"] for d in used if d.get("pitch_mm") is not None])
    nominal = spec.square_m * 1e3
    return {
        "frames_checked": len(obs),
        "frames_used": len(used),
        "distance_m": summary(dist),
        "bias_mm": summary(bias),
        "bias_by_distance": bins,
        "plane_rms_mm": summary([d["plane_rms_mm"] for d in used]),
        "plane_abs_median_mm": summary([d["plane_abs_median_mm"] for d in used]),
        "plane_abs_p90_mm": summary([d["plane_abs_p90_mm"] for d in used]),
        "pitch_mm": summary(pitch),
        "pitch_scale_vs_nominal": round(float(np.median(pitch)) / nominal, 6)
        if pitch.size
        else None,
        "nominal_square_mm": nominal,
    }


def summarize_wrist(
    obs: list[BoardObservation], spec: BoardSpec, bundle: CalibrationBundle, cfg: SensorsConfig
) -> dict[str, Any]:
    with_prior = [o for o in obs if o.T_base_cam_prior is not None]
    out: dict[str, Any] = {
        "accepted_frames": len(obs),
        "frames_with_prior": len(with_prior),
        "pnp_reproj_rms_px": summary([o.reproj_rms_px for o in obs]),
        "depth": _depth_summary(obs, spec, cfg.depth),
    }
    if not with_prior:
        out["T_base_marker_ref"] = None
        return out
    T_bm = [o.T_base_cam_prior @ o.T_cam_marker for o in with_prior if o.T_base_cam_prior]
    ref = mean_pose(T_bm)
    obj = spec.object_points()
    reproj = []
    for o in with_prior:
        assert o.T_base_cam_prior is not None
        pred = project_points(o.T_base_cam_prior.inverse() @ ref, obj, bundle.K)
        reproj.append(float(np.sqrt(np.mean(np.sum((pred - o.corners_undist_px) ** 2, axis=1)))))
    near = [o for o in with_prior if o.T_base_cam_nearest is not None]
    near_poses = [o.T_base_cam_nearest @ o.T_cam_marker for o in near if o.T_base_cam_nearest]
    out.update(
        T_base_marker_ref=_pose_json(ref),
        interpolated=_spread(T_bm, ref),
        handeye_reproj_rms_px=summary(reproj),
        nearest=None
        if not near_poses
        else {"frames": len(near_poses), **_spread(near_poses, mean_pose(near_poses))},
    )
    return out


def summarize_head(
    obs: list[BoardObservation],
    spec: BoardSpec,
    bundle: CalibrationBundle,
    wrist_ref: PoseSE3 | None,
    cfg: SensorsConfig,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accepted_frames": len(obs),
        "pnp_reproj_rms_px": summary([o.reproj_rms_px for o in obs]),
        "depth": _depth_summary(obs, spec, cfg.depth),
    }
    if not obs or bundle.T_base_head is None:
        return out
    T_bm = [bundle.T_base_head @ o.T_cam_marker for o in obs]
    ref = mean_pose(T_bm)
    out.update(T_base_marker_ref=_pose_json(ref), spread=_spread(T_bm, ref))
    if wrist_ref is not None:
        out["vs_wrist_ref"] = {
            "translation_mm": round(float(np.linalg.norm(ref.t - wrist_ref.t)) * 1e3, 4),
            "translation_base_mm": [round(float(v), 4) for v in (ref.t - wrist_ref.t) * 1e3],
            "rotation_deg": round(rotation_angle_deg(wrist_ref.R, ref.R), 5),
        }
    return out


def check_board_session(
    session_dir: Any,
    square_mm: float | None,
    cfg: SensorsConfig,
    handeye: Any = None,
) -> tuple[dict[str, Any], dict[str, CalibrationBundle]]:
    """Board report of one session + per-camera bundles with the result appended to `notes`.

    `handeye` replaces the extrinsic copied into the session. Intrinsics stay the recorded
    ones, and the session directory is not modified.
    """
    wrist = ReplaySource(session_dir, "wrist", config=cfg, handeye=handeye)
    notes = (wrist.reader.manifest.get("scenario") or {}).get("notes")
    sq = square_mm if square_mm is not None else cfg.board.square_mm
    sq = sq if sq is not None else square_mm_from_notes(notes)
    if sq is None:
        raise ValueError(
            "board square size unknown: pass --square-mm or record notes 'square_mm=<v>'"
        )
    spec = BoardSpec(cfg.board.inner_cols, cfg.board.inner_rows, sq / 1e3)
    report: dict[str, Any] = {
        "board": {"inner_cols": spec.inner_cols, "inner_rows": spec.inner_rows, "square_mm": sq},
        "handeye": {
            "source": "override" if handeye is not None else "session_copy",
            "sha256": None if handeye is None else handeye.sha256,
            "path": None if handeye is None else handeye.source_path,
        },
        "cameras": {},
    }
    bundles: dict[str, CalibrationBundle] = {}
    obs_w, rej_w, n_w = observe_stream(wrist, spec, cfg)
    w = summarize_wrist(obs_w, spec, wrist.bundle, cfg)
    w.update(frames_checked=n_w, rejected=rej_w)
    report["cameras"]["wrist"] = w
    ref = None
    if w.get("T_base_marker_ref"):
        ref = PoseSE3(np.array(w["T_base_marker_ref"]["matrix"]), "base", "marker")
    bundles["wrist"] = _annotated(wrist.bundle, _wrist_note(w))
    if "head" in wrist.reader.camera_ids:
        head = ReplaySource(session_dir, "head", config=cfg, handeye=handeye)
        obs_h, rej_h, n_h = observe_stream(head, spec, cfg)
        hrep = summarize_head(obs_h, spec, head.bundle, ref, cfg)
        hrep.update(frames_checked=n_h, rejected=rej_h)
        report["cameras"]["head"] = hrep
        bundles["head"] = _annotated(head.bundle, _head_note(hrep))
    return report, bundles


def _fmt(stats: dict[str, Any] | None, key: str = "p95") -> str:
    return "n/a" if not stats else f"{stats[key]:g}"


def _wrist_note(w: dict[str, Any]) -> str:
    sp = w.get("interpolated") or {}
    d = w.get("depth") or {}
    return (
        f"P1 S2 R-CALIB check: board-in-base spread p95 {_fmt(sp.get('translation_mm'))} mm / "
        f"{_fmt(sp.get('rotation_deg'))} deg over {w.get('frames_with_prior')} frames; hand-eye "
        f"reproj p95 {_fmt(w.get('handeye_reproj_rms_px'))} px; depth bias median "
        f"{_fmt(d.get('bias_mm'), 'median')} mm, pitch scale {d.get('pitch_scale_vs_nominal')}; "
        "dt_robot_cam_ns = 0 not estimated (ADR-0004)"
    )


def _head_note(h: dict[str, Any]) -> str:
    vs = h.get("vs_wrist_ref") or {}
    return (
        f"P1 S2 R-CALIB check: head board-in-base vs. wrist reference {vs.get('translation_mm')} mm / "
        f"{vs.get('rotation_deg')} deg over {h.get('accepted_frames')} frames"
    )


def _annotated(bundle: CalibrationBundle, note: str) -> CalibrationBundle:
    from dataclasses import fields

    kwargs = {f.name: getattr(bundle, f.name) for f in fields(bundle) if f.name != "calibration_id"}
    kwargs["notes"] = f"{bundle.notes}; {note}" if bundle.notes else note
    return CalibrationBundle.create(**kwargs)
