# P6 — BA & Pose Graph

**Dependencies:** P4 + P5
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Robotic camera prior alone does not create additional constraints on independent object motion; do not claim automatic gain.

## Task sequence

- [ ] **B1 — Factor API.** SE3 manifolds, residual direction, robust losses and covariance; Ceres backend optional Python reference.

- [ ] **B2 — Visual factors.** 2D reprojection and depth factors from long-lived landmark tracks; deduplicate correlated measurements.

- [ ] **B3 — Geometric factors.** Surface point-to-plane / model residual and robust correspondence filtering.

- [ ] **B4 — Local window.** First fix X_i=T_base_cam_robot; optimize relative object poses Z_i and object landmarks; anchor gauge.

- [ ] **B5 — Motion factor.** Base-object motion prior with real dt, weakly weighted and adaptive to abrupt motion.

- [ ] **B6 — Robot/camera optional.** V2 jointly refine X_i with measured robot covariance; require independent static background constraints when useful.

- [ ] **B7 — Global graph.** Keyframe-relative SE3 edges, loop closures, outlier switch/robust checks, marginalization strategy.

- [ ] **B8 — Benchmark.** Jacobian numeric check (analytic unit test), R-MARKER GT and R-STATIC spread, held-out surface validation; compare frame-only vs window vs robot-aided.

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
