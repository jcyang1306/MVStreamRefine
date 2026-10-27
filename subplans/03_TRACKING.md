# P3 — Fast 6D Tracking

**Dependencies:** P2
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Outputs PoseEstimate; avoid implying camera robot pose reveals arbitrary free-moving object pose.

## Task sequence

- [ ] **T1 — Reference-frame initialization.** Anchor the immutable object frame at the first accepted keyframe as defined in ADR-0005; persist `T_base_obj_init`.

- [ ] **T2 — Robot-assisted prediction.** Estimate base-frame object motion from past poses; current camera prior yields predicted T_cam_obj.

- [ ] **T3 — Pose hypotheses.** 3D-3D RANSAC+SVD for matched feature points; fallback predicted ICP, multiple hypotheses if ambiguous.

- [ ] **T4 — Geometry refinement.** Point-to-plane multiscale ICP with occlusion/visibility, normals, robust gating and conservative corrections.

- [ ] **T5 — Validity & degeneracy.** Inliers, overlap, geometric RMSE, Hessian rank, uncertainty and symmetry-aware status.

- [ ] **T6 — State machine.** INITIALIZING, TRACKING, DEGRADED, LOST, RELOCALIZING; no committed map update on failure.

- [ ] **T7 — Validation.** R-STATIC (`T_base_obj` spread), R-MARKER (RPE/ATE vs. board), R-BOTH and R-HARD (failure %, degeneracy, lost target, drift); compare against robot-prior-only and ICP-only baselines.

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
