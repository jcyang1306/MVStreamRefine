# P4 — Keyframe Memory

**Dependencies:** P3
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Keyframes stores measurements, not the only copy of geometry; do not mutate original frames after optimization. Selection is by object-frame viewpoint and covisibility (ADR-0008); no loop-closure keyframe is expected.

## Task sequence

- [ ] **K1 — Canonical keyframe.** Immutable image/depth/mask refs, pose prior, features, observation IDs, checksum, calibration and versions.

- [ ] **K2 — Selection.** Relative object viewpoint (view direction in the object frame), overlap, novelty, spatial feature distribution, uncertainty, min-frame spacing. A new keyframe is inserted when the overlap with the anchoring keyframe drops below a threshold, so the anchor stays close (ADR-0008).

- [ ] **K3 — Local window.** Deterministic bounded selection for BA/tracking: the keyframes whose object-frame viewpoints are nearest to the current predicted viewpoint and covisible with it (the anchoring set for P3 T3/T4); diversify viewpoints and avoid redundant samples.

- [ ] **K4 — Global memory.** Index by appearance, visibility signatures and object-frame viewpoint; stable IDs and lossless retrieval. Revisits inside the reachable arc (back-and-forth motion) re-anchor to existing keyframes instead of creating new ones; no loop closure is expected (ADR-0008).

- [ ] **K5 — Persistence.** Atomic files/manifest, load-restart, corruption detection and storage policy.

- [ ] **K6 — Validation.** Reproducibility of choices, novelty, retrieval recall, memory bound, immutable raw observations.

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
