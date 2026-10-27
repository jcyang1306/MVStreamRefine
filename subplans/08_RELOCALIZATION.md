# P8 — Lost Tracking & Recovery

**Dependencies:** P4 + P5 + P6 (R2 needs the object map geometry)
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
When target is fully invisible and free-moving, prediction is not a ground-truth observation; no fusion.

## Task sequence

- [ ] **R1 — Candidate retrieval.** Use descriptors/tracks and keyframe memory; avoid robot TCP-only matching.

- [ ] **R2 — Global hypotheses.** Depth-assisted RANSAC and/or model geometry, symmetry-aware multiple pose hypotheses.

- [ ] **R3 — Geometric verification.** Depth reprojection, visibility, cross-frame consistency and uncertainty tests.

- [ ] **R4 — Resume protocol.** Transition LOST->RELOCALIZING->TRACKING only on verified pose; reuse map, never restart object frame.

- [ ] **R5 — Tests.** Occlusion, out-of-FOV reentry, rotation >90deg, symmetric object and poor mask.

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
