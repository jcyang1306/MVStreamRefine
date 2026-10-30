# Coordinates, time, and observability

## Frames and composition
- `T_A_B` maps homogeneous column points from B to A: `p_A = T_A_B @ p_B`.
- Frames: `base`, `tcp`, `cam` (color optical, OpenCV axes), `obj`, `marker`, `export` (see `DATA_CONTRACTS.md`).
- `T_base_cam = T_base_tcp @ T_tcp_cam`; `T_base_obj = T_base_cam @ T_cam_obj`; `T_cam_obj = inv(T_base_cam) @ T_base_obj`.
- Backprojection: `p_cam = depth_m * inv(K) @ [u, v, 1]`, `p_obj = inv(T_cam_obj) @ p_cam`.
- Open3D TSDF in the object frame: `extrinsic = T_cam_obj` (object/world → camera). The inverse is **not** taken. The mapping wrapper is the only place that talks to Open3D extrinsics.

## Robot and calibration inputs
- RM65 controller returns TCP pose as `[x, y, z, rx, ry, rz]` (m, rad), semantics `T_base_tcp`, rotation `R = Rz(rz) @ Ry(ry) @ Rx(rx)` (= scipy `from_euler('xyz')`, extrinsic). Euler angles exist only inside the robot adapter; core state uses rotation matrices.
- Joint angles are recorded alongside the controller pose so FK can be recomputed if the controller pose is ever in doubt.
- Hand-eye file `handeye_tf.txt` (reference: 2026-10-28, ADR-0007 amendment): block `wrist_cam2` = `T_tcp_cam`; block `base_cam1` = `T_base_head`. Quality of that solve is `exploratory`. V2 algorithms stay wrist-only.
- Quaternions on disk use `xyzw` with canonical `w >= 0`.

## Object frame
Defined by ADR-0005: origin at the median of valid object points of the first accepted keyframe, axes parallel to `base`. `T_base_obj_init` is stored once and never re-centered. Exports carry `T_object_export` explicitly.

## Depth
- Registered depth and color share the color pixel grid and `K`.
- `depth_m = depth_raw_u16 * depth_units_m` (ADR-0003); `0` = invalid. Range limits and edge/hole removal are observation-stage (P2) operations with documented parameters.
- Systematic depth bias vs. distance is measured on R-CALIB / flat-target sessions before claiming dynamic accuracy.

## Time
- Canonical domain `HOST_MONOTONIC` (ns). `timestamp_ns` = estimated mid-exposure (ADR-0004).
- All raw clock values (host arrival, RealSense global/sensor/frame timestamps, exposure, robot sample host time) are persisted; mapping between domains is a replay-time function with reported residuals.
- Robot pose at frame time: rotation slerp + translation lerp between bracketing samples; valid only if gap ≤ `sync.max_interp_gap_ns` (`configs/sensors.yaml`); never extrapolate. Residual offset `dt_robot_cam_ns` comes from `CalibrationBundle`.

## Perturbation and interpolation conventions
- se(3) vector order `[ρ (translation), φ (rotation)]` = `[tx,ty,tz,rx,ry,rz]`; `Exp`/`Log` are the exact SE(3) maps (with the V matrix), not decoupled.
- Left perturbation `T = Exp(δ) · T̄`; Jacobians and covariances follow this convention.
- Robot-pose interpolation is decoupled (slerp + lerp) because controller translation and rotation are sampled independently; pose-graph interpolation (if ever needed) uses the SE(3) geodesic. Each call names which one it uses.

## Observability
- Robot pose is a **camera** pose prior; it does not measure the object's independent motion.
- With only object pixels visible, RGB-D constrains `T_cam_obj`; joint camera/object correction is weakly observable unless static background is visible. V2 first fixes `T_base_cam` from kinematics; regularized correction is optional (P6 B6).
- Symmetric objects have unobservable DOF; trackers report `SYMMETRY_AMBIGUOUS` instead of picking arbitrarily.

## Validation
- Analytic unit tests: A/B/C composition order, round-trip, frame-mismatch errors, numerical Jacobians, unit scale.
- Real-data checks: R-STATIC sessions must give near-constant `T_base_obj`; R-CALIB sessions must give a near-constant board pose in `base`.
