# P7 — Versioned Mapping & Reintegration

**Dependencies:** P5 + P6
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Optimized pose changes cannot just move an already-integrated static TSDF; rebuilding is correctness baseline.

## Task sequence

- [ ] **A1 — Pose version protocol.** Each committed keyframe has pose_version, calibration_id and map_version with idempotent updates.

- [ ] **A2 — Raw keyframes.** Preserve source RGB-D/mask at integration resolution, mask revision and TSDF parameters.

- [ ] **A3 — Full deterministic rebuild.** On optimized pose change, reset and reintegrate validated keyframes; atomic snapshot publication.

- [ ] **A4 — Tracking map.** Maintain local point/surfel/index normals in same object frame without reading mutable TSDF during edits.

- [ ] **A5 — Advanced weights optional.** Depth incidence, uncertainty and mask edge models in pluggable backend; verify actual Open3D API limitations.

- [ ] **A6 — Tests.** Rebuilt map equals clean integration at corrected poses; interrupted rebuild leaves previous good map unchanged.

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
