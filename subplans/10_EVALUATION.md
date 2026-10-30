# P10 — Benchmarks & Quality Assurance

**Dependencies:** Every phase
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Millimeter-level precision is a measured research goal, not guaranteed by 2-mm TSDF voxel or any optimizer.

## Task sequence

- [ ] **E1 — Fixtures.** Record and register the scenario set in `../INTEGRATION_PLAN.md` (R-CALIB, R-STATIC, R-MARKER, R-HANDHELD, R-BOTH × object set; R-HARD and reflective objects are out of scope, owner 2026-10-30); keep a recording checklist per session.

- [ ] **E2 — Ground truth.** R-STATIC constant-pose metric, R-MARKER board poses with fixed `T_marker_obj` alignment, shape GT object (machined part / gauge block / calibrated sphere / external scan).

- [ ] **E3 — Pose metrics.** Translation, rotation, RPE/ATE after valid fixed-gauge alignment, symmetry-aware DOFs, loss and recovery rate.

- [ ] **E4 — Geometry metrics.** Held-out view residual median/RMSE/P95, coverage of the reachable surface (ADR-0008), surface thickness and GT shape difference; avoid training-data-only self evaluation.

- [ ] **E5 — Robustness.** Timing perturbation, depth noise, blur, mask errors, frames skipped, low texture and large rotation.

- [ ] **E6 — Ablations.** ICP-only, 2D feature+ICP, frontend+BA, robot prediction, robot pose-prior joint BA, reintegration.

- [ ] **E7 — Reports.** Versioned YAML/config, revision hash, hardware, dataset, plots, metrics and pass/fail thresholds.

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
