# P2 — Observation

**Dependencies:** P1
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Does not estimate final SE3 pose; output ObjectObservation and Correspondence. Avoid using mask holes as invented depth.

## Task sequence

- [ ] **O1 — Segmentation adapter.** SAM 2.1 from reusable external module, offline mask cache, interactive initial prompt, status, confidence if available.

- [ ] **O2 — Depth preprocessing.** Depth-to-color registration validation, holes and boundary removal, pixel backprojection, noise masks and normals.

- [ ] **O3 — Feature extraction.** ORB baseline inside safe foreground; abstract SuperPoint/LightGlue and point-track adapters. Store raw pixel coordinate and descriptor.

- [ ] **O4 — Long-lived tracks.** Persistent track ID and per-frame UV, depth, status, confidence, occlusion/retirement.

- [ ] **O5 — Correspondences.** 2D-2D and 3D-3D cross-keyframe matches, geometric screening and traceable rejected outliers.

- [ ] **O6 — Quality benchmark.** On R-STATIC sessions the robot prior gives reference 3D-3D transforms between frames: match inlier precision/recall against that reference, track repeatability, depth coverage; textureless/reflective object sessions.

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
