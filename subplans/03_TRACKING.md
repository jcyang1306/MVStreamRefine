# P3 — Fast 6D Tracking

**Dependencies:** P2
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Outputs PoseEstimate; avoid implying camera robot pose reveals arbitrary free-moving object pose.

Keyframe-anchored tracking (ADR-0008): the live frame is registered against the local keyframe set / object map, not chained to the previous frame. That bounds pose error by distance to the nearest keyframe. It is **not** bundle adjustment — P6 jointly refines keyframe poses and landmarks later. Do not add BA, loop closures, or a separate "drift correction" step in this phase.

## Task sequence

- [ ] **T1 — Reference-frame initialization.** Anchor the immutable object frame at the first accepted keyframe as defined in ADR-0005; persist `T_base_obj_init`.

- [ ] **T2 — Robot-assisted prediction.** Estimate base-frame object motion from past poses; current camera prior yields predicted T_cam_obj. The previous frame is used **only** for this prediction / initialization (ADR-0008).

- [ ] **T3 — Pose hypotheses (keyframe-anchored).** Register the current frame against the local keyframe set from P4 K3 (not the previous frame): descriptor matches to each local keyframe (P2 O5 `orb`) plus short-baseline track matches to the most recent keyframe; 3D-3D RANSAC+SVD; fallback predicted ICP against the local object map; multiple hypotheses if ambiguous. Record which keyframe anchors each estimate.

- [ ] **T4 — Geometry refinement.** Point-to-plane multiscale ICP against the local object map / anchoring keyframes with occlusion/visibility, normals, robust gating and conservative corrections. Pose error must not grow with frame count while the anchoring keyframe is unchanged.

- [ ] **T5 — Validity & degeneracy.** Inliers, overlap, geometric RMSE, Hessian rank, uncertainty and symmetry-aware status.

- [ ] **T6 — State machine.** INITIALIZING, TRACKING, DEGRADED, LOST, RELOCALIZING; no committed map update on failure.

- [ ] **T7 — Validation.** R-STATIC (`T_base_obj` spread), R-MARKER (RPE/ATE vs. board), R-BOTH (failure %, degeneracy, drift); drift vs. frames-since-anchor-keyframe and vs. viewpoint distance (ADR-0008); compare keyframe-anchored vs. frame-to-frame chaining, robot-prior-only and ICP-only baselines. R-HARD is out of scope (owner 2026-10-30).

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run the matching I-gate in `../INTEGRATION_PLAN.md` by replaying registered real sessions (`fixtures/registry.yaml`); live-hardware checks are additional, marked `hardware`.
- Performance/accuracy log: include relevant latency, failure cases, quality and version numbers.

## Completion checklist
- [ ] API documentation and schema tests complete.
- [ ] Deterministic offline test passes and reproduction command recorded.
- [ ] Integration gate evidence and quantitative metrics saved.
- [ ] Known limitations recorded; no false success or hidden fallback.
- [ ] Handoff report names changed files, tests, residual risks and next phase.
