# MVStreamRefine V2 — Master Plan

> Status: PLANNED. This file is the single source of truth for system architecture and implementation rules.
> Scope: one wrist-mounted RealSense RGB-D camera and one independently moving **unknown rigid object**; camera moves with robot. No CAD prerequisite. Offline-first, real-time later. Robot pose is a **camera pose prior**, not an object pose measurement.

## 1. Mission and non-goals
- Estimate `T_cam_obj(t)` continuously, derive `T_base_obj(t)`, and reconstruct object geometry in one immutable object-reference frame.
- Track and reconstruct while camera and object move independently. Handle poor masks, occlusion, degeneracy, relocalization and trajectory correction.
- Retain original RGB-D/masks for reintegration; produce reproducible quality metrics.
- First release assumes **single rigid object**; not multiple moving objects, non-rigid bodies, active robot motion planning, or guaranteed sub-mm accuracy.
- Never claim accuracy solely from TSDF voxel size or ICP fitness.
- **Hardware (confirmed 2026-10-27):** wrist Intel RealSense **D405**, serial `323622271240`, 1280×720 @ 30 fps, depth unit `1e-4 m`, color and depth from the same imager (hardware-synchronized), optimal range ≈ 7–50 cm; RealMan RM65 at `192.168.1.19:8080`, SDK `Robotic_Arm==1.1.6`; reference hand-eye = 2026-10-12 calibration (`wrist_cam2 = T_tcp_cam`, ADR-0007). Fixed head RealSense D435 (serial `244622070961`, `base_cam1 = T_base_head`) is recorded as an optional stream; V2 algorithms stay wrist-only until a later ADR.
- **V1 reference:** `ref/MVStreamRefine` (read-only; static-object TSDF pipeline, RealSense/RM65 capture, SAM 2.1 inference package, ICP). Reuse only through explicit adapters with parity tests (ADR-0006).
- **Validation data:** real sessions newly recorded with the V2 recorder only; no synthetic datasets (ADR-0002) and no V1 data reuse (ADR-0007). Available GT aid: checkerboard with 11×8 inner corners (square size to be measured).
- **Runtime:** workstation `CW20175` via Cursor Remote-SSH, Docker only (ADR-0006); data under `./data`; V1 at `ref/MVStreamRefine` (symlink allowed).

## 2. Authority and hierarchy
1. `MASTER_PLAN.md` (scope, dependency boundaries, gates)
2. `contracts/*.md` (data, math, API, persistency)
3. `subplans/XX_*.md` (implementation details limited to owning module)
4. `INTEGRATION_PLAN.md` (cross-module assembly and verification)
5. `decisions/ADR-*.md` (approved exceptions / durable choices)
Conflicts are resolved by updating the master plan/contract in a separate architecture change, **not** by silently changing another module.

## 3. Immutable engineering conventions
- `T_A_B` maps coordinates in B to A. `T_base_cam = T_base_tcp @ T_tcp_cam`; `T_cam_obj = inverse(T_base_cam) @ T_base_obj`.
- Backprojection `p_obj = inverse(T_cam_obj) @ p_cam`; Open3D TSDF volume in object frame uses `extrinsic=T_cam_obj` (world/object -> camera). Unit: **meters**; depth PNG stores raw `uint16` device units with `depth_units_m` in the manifest (wrist D405 `1e-4`; head D435 `1e-3`), converted to meters exactly once (ADR-0003); time `HOST_MONOTONIC` **nanoseconds**, mid-exposure, with all raw clocks persisted (ADR-0004).
- RGB-D is registered/calibrated; intrinsic for the actual registered image resolution. Quaternion in external datasets: `xyzw`, internally rotation in `SO(3)`; no Euler in core state (RM65 `xyzrpy` is converted inside the robot adapter only).
- `T_cam_obj` poses all refer to a **single immutable object frame** fixed at initialization: origin at the first accepted keyframe's object-point median, axes parallel to `base` (ADR-0005); optional model-centric export uses a separate constant transform.
- Tracking, optimization and mapping **must** be independent; no global mutable pose map accessed across threads.
- `frame_id`, `timestamp_ns`, `session_id`, `calibration_id`, `pose_version`, `map_version` accompany all asynchronous state-bearing messages.
- Every algorithm provides uncertainty/degeneracy status rather than always returning a valid six-DOF pose.
- No silent frame dropping, pose extrapolation, or committed fusion on uncertain poses.

## 4. Architecture & module ownership
- core: transform math, typed messages, configuration and errors.
- sensors: acquisition, calibration loading, clock alignment and interpolation. No SAM/ICP.
- observation: segmentation, preprocessing, feature tracks and correspondence generation. Does not output final object pose.
- tracking: fast pose prediction, hypothesis generation, geometric refinement and validation.
- keyframes: selection, short/long-term retrieval and durable measurement store.
- mapping: object-coordinate tracking geometry, TSDF integration, committed state and rebuild.
- optimization: factors, local BA, global pose graph and optional robot pose correction; no direct TSDF mutation.
- relocalization: retrieve, hypothesize, verify and resume same object map.
- pipelines: workflow composition and async versioned event orchestration only.
- evaluation: independent pose/shape metrics and reproducibility.

## 5. Releases and dependencies
| Phase | Owned plan | Depends on | Runnable gate |
|---|---|---|---|
| P0 | `subplans/00_FOUNDATION.md` | none | unit-tested SE(3), contract types, session IO, `mvsr record` (D405 + D435 + RM65); one owner-recorded session passes `verify-session --p0-gate` |
| P1 | `subplans/01_SENSORS.md` | P0 | replay to `SensorFrame` with clock mapping + interpolated robot poses; timing + hand-eye report |
| P2 | `subplans/02_OBSERVATION.md` | P1 | masks, keypoint tracks and grounded 2D/3D matches |
| P3 | `subplans/03_TRACKING.md` | P2 | moving-object relative pose on R-STATIC, R-MARKER and R-BOTH recordings |
| P4 | `subplans/04_KEYFRAMES.md` | P3 | deterministic, persisted keyframe memory |
| P5 | `subplans/05_MAPPING_MVP.md` | P3–P4 | recognizable object-centric fusion on dynamic sequence |
| P6 | `subplans/06_OPTIMIZATION.md` | P4–P5 | corrected keyframe poses, improved held-out residuals |
| P7 | `subplans/07_MAPPING_ADVANCED.md` | P5–P6 | model rebuilt correctly after historical pose corrections |
| P8 | `subplans/08_RELOCALIZATION.md` | P4, P5, P6 | recovery following tracking loss without new map |
| P9 | `subplans/09_REALTIME.md` | P3–P8 | stable bounded-memory streaming and shutdown |
| P10 | `subplans/10_EVALUATION.md` | continuously | benchmark report / regression gate |

## 6. Architectural constraints for agents
- Read this file, `contracts/*`, and only the target subplan before editing.
- Report which files will change; do not rewrite unrelated modules or old static pipeline without explicit plan change.
- Each milestone: implement -> format/typecheck -> unit tests -> deterministic integration fixture -> report measured outcome -> update checklist.
- Baseline CI replays recorded real sessions from `fixtures/registry.yaml` and cached mask revisions, so it needs no live RealSense, robot, SAM weights or GPU. Live-hardware tests are `@pytest.mark.hardware`. SDK test doubles may cover error paths only, never accuracy (ADR-0002).
- Schema migrations explicitly versioned. No handwritten, incompatible data representations.
- Preserve the existing project as read-only reference; V2 clean implementation may reuse verified algorithms via explicit adapters, never implicit imports of old globals.
- Configuration values and thresholds must be documented, with source, units and failure behavior; placeholders must be labeled.
- If a stage fails quality gates, it cannot increment committed map version.

## 7. Definition of Done for each subplan
- Public API described with units, transform directions and errors.
- Analytic unit tests and recorded real-session fixture tests repeatably pass; invariants covered.
- Core logs include per-frame identifiers, metrics, reason codes and timings.
- No unbounded queues; error recovery deterministic.
- Updated docs, a short example and a handoff report stating files changed, tests run, numerical results, known risks, and next module dependency.

## 8. Quality gates (development targets, not guarantees)
- Static calibration and depth characteristics measured before declaring dynamic accuracy.
- Start with proposed 1–2 mm *held-out surface consistency* goal on a specified object/material/distance/motion envelope; separate from **true shape error** using a reference object/scan, and separate from 6D pose accuracy with independent GT (R-STATIC constant `T_base_obj`, R-MARKER board poses; ADR-0002).
- Record median/RMSE/P95, pose failure rates, rotational errors, trajectory drift, latency and coverage; baseline every change against ICP-only and front-end only.
- Timestamps synchronized/interpolated; report residual temporal uncertainty. No claim that 50ms proximity is sufficient for precision dynamic fusion.

## 9. Change management
- Each subplan keeps checklist and artifacts.
- New external dependency must be captured in ADR, environment lock file and fallback path.
- Update integration fixtures at the boundary, not by bypassing contracts.
- Release gate needs evidence from `INTEGRATION_PLAN.md` and evaluation reports.
