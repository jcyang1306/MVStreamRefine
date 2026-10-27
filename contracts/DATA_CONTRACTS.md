# Data and API Contracts (V2, schema_version 1)

Units are meters, radians and nanoseconds unless the field name says otherwise (`_px`, `_ms`, `_us`, `_u16`). Frame names are fixed: `base` (robot base), `tcp` (RM65 tool frame as reported by the controller), `cam` (wrist D405 color optical frame, OpenCV axes x right / y down / z forward), `head` (fixed D435 color optical frame, same axis convention; recorded, not yet used by algorithms — ADR-0007), `obj` (immutable object frame, ADR-0005), `marker` (optional GT board), `export` (model-centric export frame). On-disk layout: `SESSION_FORMAT.md`.

## Identifiers
- `session_id: str` — `YYYYMMDDTHHMMSS_<host>_<4hex>`; one per recording. Processing runs get `run_id` of the same form.
- `camera_id: str` — `wrist` | `head`.
- `frame_id: int` — 0-based, strictly increasing per `(session_id, camera_id)` stream, assigned at camera frame arrival **including frames later dropped**; never renumbered. Frames of different cameras are related only through timestamps.
- `keyframe_id: int` — 0-based, strictly increasing per run; never reused.
- `calibration_id: str` — first 16 hex of sha256 over the canonical JSON of `CalibrationBundle` (excluding `calibration_id`).
- `pose_version: int` — per keyframe; 0 = tracking (provisional) pose; +1 per accepted `PoseUpdate`.
- `map_version: int` — per run; +1 per published `MapSnapshot`; 0 = empty map.
- `mask_revision: str` — `<producer>@<params-hash>`, e.g. `sam2.1-tiny@3f9a…`.

## Enumerations
- `ClockDomain = HOST_MONOTONIC | RS_DEVICE | RS_GLOBAL | ROBOT_CONTROLLER`.
- `SyncStatus = INTERPOLATED | GAP_TOO_LARGE | OUT_OF_RANGE | NO_ROBOT`.
- `PoseStatus = VALID | DEGRADED | INVALID`.
- `TrackState = INITIALIZING | TRACKING | DEGRADED | LOST | RELOCALIZING`.
- `KeyframeStatus = PROVISIONAL | COMMITTED | REJECTED | SUPERSEDED`.
- `ReasonCode` (string enum, extensible only via contract change): `MASK_TOO_SMALL, DEPTH_COVERAGE_LOW, POSE_SYNC_FAILED, FEW_FEATURES, LOW_INLIERS, LOW_OVERLAP, HIGH_RMSE, DEGENERATE_HESSIAN, SYMMETRY_AMBIGUOUS, LARGE_CORRECTION, PREDICTION_ONLY, STALE_VERSION, CHECKSUM_MISMATCH`.

## Common header
`Header(schema_version: int, session_id, camera_id, frame_id, timestamp_ns: int, clock_domain: ClockDomain, calibration_id)` is embedded in `SensorFrame`, `ObjectObservation`, `PoseEstimate`, `Keyframe`. `timestamp_ns` is the estimated **mid-exposure** time (ADR-0004).

## Primitives
`PoseSE3(matrix: float64[4,4], dst: str, src: str, covariance: float64[6,6] | None = None)`
- Validates finite values, `R^T R ≈ I` (atol 1e-6), `det R ≈ +1`, last row `[0,0,0,1]`, non-empty frame names.
- `a @ b` requires `a.src == b.dst`; result frames `(a.dst, b.src)`. `inverse()` swaps frames. Mismatched frames raise `FrameMismatchError`.
- Covariance: tangent order `[tx,ty,tz,rx,ry,rz]`, **left** perturbation `T = Exp(δ) · T̄`, so δ is expressed in `dst`. Inversion/composition propagate covariance with the adjoint; never copy covariance across frames unchanged.

`CalibrationBundle(calibration_id, camera_id, camera_serial, camera_model, mount: "wrist" | "fixed", width, height, K: float64[3,3], distortion_model: str, distortion: float64[5], depth_units_m: float, T_tcp_cam: PoseSE3(dst="tcp", src="cam") | None, T_base_head: PoseSE3(dst="base", src="head") | None, dt_robot_cam_ns: int = 0, source_files: dict[str, sha256], notes: str)`.
Exactly one extrinsic is set: `T_tcp_cam` for `mount="wrist"`, `T_base_head` for `mount="fixed"`. `K`/distortion are for the **registered color** image at `width×height`, as reported by the SDK at record time.

`RobotSample(seq: int, t_host_ns: int, clock_domain, T_base_tcp: PoseSE3(base,tcp), joints_rad: float64[6] | None, raw_xyzrpy: float64[6] | None, source: "udp_push" | "poll", read_latency_ns: int | None)`.

`FrameTiming(t_host_arrival_ns, rs_timestamp_ms: float | None, rs_timestamp_domain: str | None, rs_sensor_timestamp_us: int | None, rs_frame_timestamp_us: int | None, exposure_us: float | None, rs_frame_number: int | None, method: str)` — `method` states how `Header.timestamp_ns` was derived.

`PoseSyncInfo(status: SyncStatus, method: str, gap_before_ns: int | None, gap_after_ns: int | None, robot_seq_before: int | None, robot_seq_after: int | None)`.

## Messages
`SensorFrame(header, color_rgb_u8: uint8[H,W,3], depth_raw_u16: uint16[H,W], depth_units_m: float, depth_m_f32: float32[H,W] (derived once; 0.0 = invalid), valid_depth_mask: bool[H,W], K: float64[3,3], T_base_tcp: PoseSE3 | None, T_base_cam_prior: PoseSE3(base,cam) | None, pose_sync: PoseSyncInfo, timing: FrameTiming, source_meta: dict)`.
Frames whose sync failed are still emitted with `T_base_cam_prior=None` and the failing `SyncStatus`; consumers decide, sources never skip silently.

`ObjectObservation(header, instance_id: int, mask_revision, mask_binary: bool[H,W], mask_prob: float32[H,W] | None, geometry_mask: bool[H,W], pixel_uv: int32[N,2], points_cam_m: float32[N,3], normals_cam: float32[N,3] | None, keypoints_uv: float32[M,2], keypoint_depth_m: float32[M] (NaN = invalid), descriptors: ndarray[M,D], descriptor_type: str, track_ids: int64[M] (-1 = untracked), quality: dict)`.
`geometry_mask` = pixels allowed to produce 3D points (mask after erosion, depth validity and range limits). `quality` holds at least `mask_area_px`, `valid_depth_ratio`, `n_keypoints`, `mask_score` (if the segmenter provides one).

`CorrespondenceSet(src_frame_id, tgt_frame_id, src_uv: float32[K,2], tgt_uv: float32[K,2], src_point_cam_m: float32[K,3], tgt_point_cam_m: float32[K,3] (NaN rows = no depth), sigma_px: float32[K], sigma_m: float32[K], track_ids: int64[K], method: str, inlier: bool[K], rejection_reason: list[str | None])`.
`sigma_px` is isotropic pixel std-dev in each image; `sigma_m` is isotropic 3D std-dev in each frame's `cam`. Rejected matches are kept with a reason, not deleted.

`TrackingReferences(prev: PoseEstimate | None, predicted_T_cam_obj: PoseSE3(cam,obj) | None, local_keyframes: tuple[KeyframeView, ...])`; `KeyframeView(keyframe_id, observation: ObjectObservation, T_cam_obj: PoseSE3, pose_version)` — read-only.

`MapView(map_version, points_obj_m: float32[P,3], normals_obj: float32[P,3], colors_u8: uint8[P,3] | None, source_keyframes: dict[keyframe_id, pose_version])` — immutable (arrays non-writeable).

`PoseEstimate(header, T_cam_obj: PoseSE3(cam,obj) | None, T_base_obj: PoseSE3(base,obj) | None, T_base_cam_used: PoseSE3(base,cam) | None, status: PoseStatus, track_state: TrackState, reason_codes: tuple[ReasonCode, ...], quality: dict, source: str, pose_version: int = 0)`.
`status == INVALID` ⇔ `T_cam_obj is None`. Identity is never a failure default. `quality` holds at least `inliers`, `overlap`, `rmse_m`, `hessian_min_eig`, `n_hypotheses`, `correction_from_prediction_m/deg`.

`Keyframe(keyframe_id, header, observation_ref: str, mask_revision, raw_refs: dict[str, (path, sha256)], T_base_cam_prior: PoseSE3 | None, T_cam_obj_provisional: PoseSE3, T_cam_obj_current: PoseSE3, pose_version: int, feature_track_ids: int64[M], status: KeyframeStatus, checksum: str)`.

`PoseUpdate(session_id, run_id, keyframe_id, frame_id, old_version, new_version, T_cam_obj: PoseSE3(cam,obj), covariance: float64[6,6] | None, optimizer_run_id, created_ns)`.
Applying an update whose `old_version` ≠ current version is rejected with `STALE_VERSION`; re-applying the same `(keyframe_id, new_version)` is a no-op (idempotent).

`MapSnapshot(session_id, run_id, map_version, parent_map_version, T_base_obj_init: PoseSE3(base,obj), calibration_id, committed: dict[keyframe_id, pose_version], mask_revision, tsdf_params_hash, geometry_paths: dict[str, str], created_ns)`.

## Validation rules
- Depth is converted from raw units to meters exactly once (session reader / sensor source). No other module scales depth.
- All `Header.timestamp_ns` values of one run are in `HOST_MONOTONIC`; raw values from other domains live only in `FrameTiming` / `RobotSample`.
- Published arrays are set non-writeable; a revised mask is a new `mask_revision`, never an in-place edit.
- Metrics never mix `obj` and `base` coordinates; every metric record names its frame.
- Transforms are covered by round-trip and frame-mismatch tests.

## Serialization
- JSON metadata: UTF-8, sorted keys, `schema_version` field in every top-level object, matrices as row-major nested lists, floats written with `repr` precision.
- Arrays: NPZ (uncompressed for speed unless stated). Images: PNG (`SESSION_FORMAT.md`).
- Every persisted file has a sha256 recorded in its parent manifest.
- Atomic writes: write `*.tmp` → `fsync` → `rename` → manifest update. Readers verify checksums and raise `ChecksumMismatchError`.

## Public ports
- `SensorSource.next() -> SensorFrame | EndOfStream`
- `ObservationBuilder.process(SensorFrame) -> ObjectObservation`
- `ObjectTracker.track(SensorFrame, ObjectObservation, TrackingReferences, MapView | None) -> PoseEstimate`
- `KeyframeManager.consider(ObjectObservation, PoseEstimate) -> Keyframe | None`
- `Optimizer.submit(Keyframe, CorrespondenceSet) -> None`; `Optimizer.poll_pose_updates() -> list[PoseUpdate]`
- `ObjectMap.commit(Keyframe, PoseEstimate) -> int (map_version)`; `ObjectMap.apply_pose_updates(list[PoseUpdate]) -> int`; `ObjectMap.snapshot() -> MapSnapshot`; `ObjectMap.view() -> MapView`
