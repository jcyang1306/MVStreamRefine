"""P3 keyframe-anchored 6D object tracker.

The previous frame is used only by `predict`; pose hypotheses are measured against immutable
keyframe observations (ADR-0008). BA, loop closure and map commits are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mvsr.core.errors import MvsrError
from mvsr.core.se3 import PoseSE3, se3_exp, se3_log, so3_log
from mvsr.core.types import (
    CalibrationBundle,
    CorrespondenceSet,
    Header,
    KeyframeView,
    MapView,
    ObjectObservation,
    PoseEstimate,
    PoseStatus,
    ReasonCode,
    SensorFrame,
    TrackingReferences,
    TrackState,
)
from mvsr.observation.config import ObservationConfig
from mvsr.observation.correspondence import KeypointView, match_observations
from mvsr.tracking.config import TrackingConfig
from mvsr.tracking.icp import IcpResult, point_to_plane_icp


class TrackingError(MvsrError):
    """Tracking input or state violates the P3 contract."""


def _same_header(a: Header, b: Header) -> bool:
    return (
        a.schema_version == b.schema_version
        and a.session_id == b.session_id
        and a.camera_id == b.camera_id
        and a.frame_id == b.frame_id
        and a.timestamp_ns == b.timestamp_ns
        and a.clock_domain is b.clock_domain
        and a.calibration_id == b.calibration_id
    )


@dataclass(frozen=True)
class _Hypothesis:
    pose: PoseSE3
    anchor_id: int
    method: str
    inliers: int
    candidates: int
    feature_rmse_m: float | None
    icp: IcpResult | None = None


def _empty_quality() -> dict[str, Any]:
    return {
        "inliers": 0,
        "overlap": 0.0,
        "rmse_m": None,
        "hessian_min_eig": 0.0,
        "n_hypotheses": 0,
        "correction_from_prediction_m": 0.0,
        "correction_from_prediction_deg": 0.0,
    }


def _pose_delta(a: PoseSE3, b: PoseSE3) -> tuple[float, float]:
    d = a.matrix @ np.linalg.inv(b.matrix)
    return float(np.linalg.norm(d[:3, 3])), float(np.degrees(np.linalg.norm(so3_log(d[:3, :3]))))


def _feature_rmse(cs: CorrespondenceSet, R: np.ndarray, t: np.ndarray) -> float | None:
    src = cs.src_point_cam_m
    tgt = cs.tgt_point_cam_m
    inlier = cs.inlier
    both = inlier & np.isfinite(src[:, 0]) & np.isfinite(tgt[:, 0])
    if not both.any():
        return None
    residual = src[both].astype(np.float64) @ R.T + t - tgt[both].astype(np.float64)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _model_from_keyframes(
    keyframes: tuple[KeyframeView, ...], maximum: int
) -> tuple[np.ndarray, np.ndarray] | None:
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    each = max(1, maximum // max(1, len(keyframes)))
    for kf in keyframes:
        obs = kf.observation
        if obs.normals_cam is None or not len(obs.points_cam_m):
            continue
        idx = (
            np.arange(len(obs.points_cam_m), dtype=np.intp)
            if len(obs.points_cam_m) <= each
            else np.linspace(0, len(obs.points_cam_m) - 1, each, dtype=np.intp)
        )
        T_obj_cam = kf.T_cam_obj.inverse()
        points.append(T_obj_cam.transform_points(obs.points_cam_m[idx]))
        normals.append(obs.normals_cam[idx].astype(np.float64) @ T_obj_cam.R.T)
    if not points:
        return None
    return np.concatenate(points), np.concatenate(normals)


class ObjectTracker:
    """Implementation of the `ObjectTracker.track(...) -> PoseEstimate` public port."""

    def __init__(
        self,
        bundle: CalibrationBundle,
        cfg: TrackingConfig,
        observation_cfg: ObservationConfig,
    ) -> None:
        if bundle.camera_id != "wrist":
            raise TrackingError("P3 currently supports the wrist camera only")
        self.bundle = bundle
        self.cfg = cfg
        self.observation_cfg = observation_cfg
        self.T_base_obj_init: PoseSE3 | None = None
        self.track_state = TrackState.INITIALIZING
        self._bad_frames = 0
        self._base_history: list[tuple[int, PoseSE3]] = []
        self._last_frame_id: int | None = None
        self._session_id: str | None = None

    def reset(self) -> None:
        """Clear all session-local state so this tracker can start a new deterministic replay."""
        self.T_base_obj_init = None
        self.track_state = TrackState.INITIALIZING
        self._bad_frames = 0
        self._base_history.clear()
        self._last_frame_id = None
        self._session_id = None

    def predict(self, frame: SensorFrame, previous: PoseEstimate | None) -> PoseSE3 | None:
        """Constant-velocity object prediction in `base`, converted using the current robot prior."""
        T_base_cam = frame.T_base_cam_prior
        if T_base_cam is None or previous is None or previous.T_base_obj is None:
            return None
        predicted_base = previous.T_base_obj
        if len(self._base_history) >= 2:
            (t0, p0), (t1, p1) = self._base_history[-2:]
            dt = t1 - t0
            future = frame.header.timestamp_ns - t1
            if dt > 0 and 0 <= future <= 2 * dt:
                velocity_step = p1.matrix @ np.linalg.inv(p0.matrix)
                xi = se3_log(velocity_step)
                # Do not amplify a noisy or discontinuous motion estimate.
                if np.linalg.norm(xi[:3]) <= 0.03 and np.linalg.norm(xi[3:]) <= np.radians(10):
                    M = se3_exp((future / dt) * xi) @ p1.matrix
                    predicted_base = PoseSE3(M, "base", "obj")
        return T_base_cam.inverse() @ predicted_base

    def _initialize(self, frame: SensorFrame, obs: ObjectObservation) -> PoseEstimate:
        q = _empty_quality()
        reasons: list[ReasonCode] = []
        if len(obs.points_cam_m) < self.cfg.initialization.min_geometry_points:
            reasons.append(ReasonCode.DEPTH_COVERAGE_LOW)
        if (
            int(np.isfinite(obs.keypoint_depth_m).sum())
            < self.cfg.initialization.min_keypoints_with_depth
        ):
            reasons.append(ReasonCode.FEW_FEATURES)
        if frame.T_base_cam_prior is None:
            reasons.append(ReasonCode.POSE_SYNC_FAILED)
        if reasons:
            return self._invalid(frame, reasons, q)
        assert frame.T_base_cam_prior is not None
        points_base = frame.T_base_cam_prior.transform_points(obs.points_cam_m)
        center = np.median(points_base, axis=0)
        self.T_base_obj_init = PoseSE3.from_rt(np.eye(3), center, "base", "obj")
        T_cam_obj = frame.T_base_cam_prior.inverse() @ self.T_base_obj_init
        q.update(
            inliers=int(np.isfinite(obs.keypoint_depth_m).sum()),
            overlap=1.0,
            rmse_m=0.0,
            hessian_min_eig=0.0,
            n_hypotheses=1,
            anchor_keyframe_id=-1,
        )
        pose = PoseEstimate(
            header=frame.header,
            T_cam_obj=T_cam_obj,
            T_base_obj=self.T_base_obj_init,
            T_base_cam_used=frame.T_base_cam_prior,
            status=PoseStatus.VALID,
            track_state=TrackState.INITIALIZING,
            reason_codes=(),
            quality=q,
            source="gauge_initialization",
        )
        self.track_state = TrackState.TRACKING
        self._remember(pose)
        return pose

    def _hypotheses(self, obs: ObjectObservation, refs: TrackingReferences) -> list[_Hypothesis]:
        out: list[_Hypothesis] = []
        recent_id = max((k.keyframe_id for k in refs.local_keyframes), default=-1)
        keyframes = refs.local_keyframes[-self.cfg.hypothesis.max_local_keyframes :]
        for kf in keyframes:
            methods = ["orb"]
            if self.cfg.hypothesis.use_recent_tracks and kf.keyframe_id == recent_id:
                methods.insert(0, "track")
            for method in methods:
                cs, diag = match_observations(
                    KeypointView.of(kf.observation),
                    KeypointView.of(obs),
                    self.bundle,
                    self.observation_cfg.correspondence,
                    self.observation_cfg.depth,
                    method,
                )
                if (
                    diag.rigid_R is None
                    or diag.rigid_t is None
                    or diag.inliers < self.cfg.hypothesis.min_inliers
                ):
                    continue
                relative = PoseSE3.from_rt(diag.rigid_R, diag.rigid_t, "cam", "cam")
                pose = relative @ kf.T_cam_obj
                out.append(
                    _Hypothesis(
                        pose=pose,
                        anchor_id=kf.keyframe_id,
                        method=method,
                        inliers=diag.inliers,
                        candidates=diag.candidates,
                        feature_rmse_m=_feature_rmse(cs, diag.rigid_R, diag.rigid_t),
                    )
                )
        out.sort(
            key=lambda h: (
                -h.inliers,
                float("inf")
                if refs.predicted_T_cam_obj is None
                else _pose_delta(h.pose, refs.predicted_T_cam_obj)[0],
                h.anchor_id,
                h.method,
            )
        )
        return out

    def _refine(
        self,
        hypotheses: list[_Hypothesis],
        obs: ObjectObservation,
        refs: TrackingReferences,
        map_view: MapView | None,
    ) -> list[_Hypothesis]:
        if not self.cfg.icp.enabled or obs.normals_cam is None:
            return hypotheses
        if map_view is not None:
            model: tuple[np.ndarray, np.ndarray] | None = (
                map_view.points_obj_m.astype(np.float64),
                map_view.normals_obj.astype(np.float64),
            )
        else:
            model = _model_from_keyframes(refs.local_keyframes, self.cfg.icp.max_points)
        if model is None:
            return hypotheses
        seeds = hypotheses[: self.cfg.hypothesis.max_icp_hypotheses]
        if not seeds and refs.predicted_T_cam_obj is not None:
            seeds = [
                _Hypothesis(
                    refs.predicted_T_cam_obj,
                    -1,
                    "predicted_icp",
                    0,
                    0,
                    None,
                )
            ]
        out: list[_Hypothesis] = []
        for h in seeds:
            result = point_to_plane_icp(
                model[0],
                model[1],
                obs.points_cam_m,
                obs.normals_cam,
                h.pose,
                self.cfg.icp,
            )
            if result is not None and (
                result.correction_m > self.cfg.icp.max_refinement_m
                or result.correction_deg > self.cfg.icp.max_refinement_deg
            ):
                result = None
            out.append(
                h
                if result is None
                else _Hypothesis(
                    result.T_cam_obj,
                    h.anchor_id,
                    f"{h.method}+icp",
                    h.inliers,
                    h.candidates,
                    h.feature_rmse_m,
                    result,
                )
            )
        return out + hypotheses[len(seeds) :]

    def _invalid(
        self,
        frame: SensorFrame,
        reasons: list[ReasonCode],
        quality: dict[str, Any],
    ) -> PoseEstimate:
        self._bad_frames += 1
        state = (
            TrackState.INITIALIZING
            if self.T_base_obj_init is None
            else (
                TrackState.LOST
                if self._bad_frames >= self.cfg.validity.lost_after_frames
                else TrackState.DEGRADED
            )
        )
        self.track_state = state
        return PoseEstimate(
            header=frame.header,
            T_cam_obj=None,
            T_base_obj=None,
            T_base_cam_used=frame.T_base_cam_prior,
            status=PoseStatus.INVALID,
            track_state=state,
            reason_codes=tuple(dict.fromkeys(reasons)),
            quality=quality,
            source="none",
        )

    def _remember(self, pose: PoseEstimate) -> None:
        if pose.status is PoseStatus.VALID and pose.T_base_obj is not None:
            self._base_history.append((pose.header.timestamp_ns, pose.T_base_obj))
            self._base_history = self._base_history[-2:]

    def track(
        self,
        frame: SensorFrame,
        observation: ObjectObservation,
        references: TrackingReferences,
        map_view: MapView | None = None,
    ) -> PoseEstimate:
        if not _same_header(frame.header, observation.header):
            raise TrackingError("frame and observation headers differ")
        if frame.header.calibration_id != self.bundle.calibration_id:
            raise TrackingError("frame calibration differs from tracker bundle")
        if self._session_id is None:
            self._session_id = frame.header.session_id
        elif frame.header.session_id != self._session_id:
            raise TrackingError("tracker cannot mix sessions; call reset()")
        if self._last_frame_id is not None and frame.header.frame_id <= self._last_frame_id:
            raise TrackingError(
                f"frames must increase ({frame.header.frame_id} after {self._last_frame_id})"
            )
        self._last_frame_id = frame.header.frame_id
        for keyframe in references.local_keyframes:
            header = keyframe.observation.header
            if (
                header.session_id != frame.header.session_id
                or header.camera_id != frame.header.camera_id
                or header.calibration_id != frame.header.calibration_id
            ):
                raise TrackingError("keyframe observation is incompatible with current frame")
        if map_view is not None:
            stale = any(
                keyframe.keyframe_id in map_view.source_keyframes
                and map_view.source_keyframes[keyframe.keyframe_id] != keyframe.pose_version
                for keyframe in references.local_keyframes
            )
            if stale:
                return self._invalid(frame, [ReasonCode.STALE_VERSION], _empty_quality())
        if self.T_base_obj_init is None:
            return self._initialize(frame, observation)
        q = _empty_quality()
        reasons: list[ReasonCode] = []
        if not observation.mask_binary.any():
            reasons.append(ReasonCode.MASK_TOO_SMALL)
        if len(observation.points_cam_m) < self.cfg.initialization.min_geometry_points:
            reasons.append(ReasonCode.DEPTH_COVERAGE_LOW)
        if not references.local_keyframes and not (
            map_view is not None and references.predicted_T_cam_obj is not None
        ):
            reasons.append(ReasonCode.LOW_INLIERS)
        if reasons:
            return self._invalid(frame, reasons, q)
        hypotheses = self._hypotheses(observation, references)
        refined = self._refine(hypotheses, observation, references, map_view)
        if not refined:
            if (
                references.predicted_T_cam_obj is None
                or self._bad_frames >= self.cfg.validity.prediction_only_frames
            ):
                return self._invalid(frame, [ReasonCode.LOW_INLIERS], q)
            refined = [
                _Hypothesis(
                    references.predicted_T_cam_obj,
                    -1,
                    "prediction_only",
                    0,
                    0,
                    None,
                )
            ]
            reasons.append(ReasonCode.PREDICTION_ONLY)

        def rank(hypothesis: _Hypothesis) -> tuple[float, int, float, int, int]:
            prediction_distance = 0.0
            if references.predicted_T_cam_obj is not None:
                distance_m, distance_deg = _pose_delta(
                    hypothesis.pose, references.predicted_T_cam_obj
                )
                prediction_distance = distance_m + 0.002 * distance_deg
            return (
                prediction_distance,
                0 if hypothesis.icp is not None else 1,
                float("inf") if hypothesis.icp is None else hypothesis.icp.rmse_m,
                -hypothesis.inliers,
                hypothesis.anchor_id,
            )

        refined.sort(key=rank)
        best = refined[0]
        if len(refined) > 1:
            alternative = refined[1]
            alternative_m, alternative_deg = _pose_delta(best.pose, alternative.pose)
            if (
                alternative.inliers >= 0.9 * max(1, best.inliers)
                and alternative_m <= 0.03
                and alternative_deg >= 15.0
            ):
                reasons.append(ReasonCode.SYMMETRY_AMBIGUOUS)
        overlap = best.inliers / max(1, best.candidates) if best.icp is None else best.icp.overlap
        rmse = best.feature_rmse_m if best.icp is None else best.icp.rmse_m
        hessian = 0.0 if best.icp is None else best.icp.hessian_min_eig
        correction_m = correction_deg = 0.0
        if references.predicted_T_cam_obj is not None:
            correction_m, correction_deg = _pose_delta(best.pose, references.predicted_T_cam_obj)
        q.update(
            inliers=best.inliers if best.icp is None else best.icp.correspondences,
            overlap=overlap,
            rmse_m=rmse,
            hessian_min_eig=hessian,
            n_hypotheses=len(refined),
            correction_from_prediction_m=correction_m,
            correction_from_prediction_deg=correction_deg,
            anchor_keyframe_id=best.anchor_id,
            method=best.method,
            icp_correction_m=None if best.icp is None else best.icp.correction_m,
            icp_correction_deg=None if best.icp is None else best.icp.correction_deg,
        )
        if best.inliers < self.cfg.hypothesis.min_inliers and best.icp is None:
            reasons.append(ReasonCode.LOW_INLIERS)
        if overlap < self.cfg.validity.min_overlap:
            reasons.append(ReasonCode.LOW_OVERLAP)
        if rmse is not None and rmse > self.cfg.validity.max_rmse_m:
            reasons.append(ReasonCode.HIGH_RMSE)
        if best.icp is not None and hessian < self.cfg.validity.min_hessian_eig:
            reasons.append(ReasonCode.DEGENERATE_HESSIAN)
        if (
            correction_m > self.cfg.validity.max_correction_m
            or correction_deg > self.cfg.validity.max_correction_deg
        ):
            reasons.append(ReasonCode.LARGE_CORRECTION)
        degraded = bool(reasons)
        status = PoseStatus.DEGRADED if degraded else PoseStatus.VALID
        previous_state = self.track_state
        if degraded:
            self._bad_frames += 1
            state = (
                TrackState.LOST
                if self._bad_frames >= self.cfg.validity.lost_after_frames
                else TrackState.DEGRADED
            )
        else:
            self._bad_frames = 0
            state = (
                TrackState.RELOCALIZING
                if previous_state is TrackState.LOST
                else TrackState.TRACKING
            )
        self.track_state = state
        T_base_obj = None if frame.T_base_cam_prior is None else frame.T_base_cam_prior @ best.pose
        pose = PoseEstimate(
            header=frame.header,
            T_cam_obj=best.pose,
            T_base_obj=T_base_obj,
            T_base_cam_used=frame.T_base_cam_prior,
            status=status,
            track_state=state,
            reason_codes=tuple(dict.fromkeys(reasons)),
            quality=q,
            source=best.method,
        )
        self._remember(pose)
        return pose
