# P5 — Object-centric Mapping MVP

**Dependencies:** P3 + P4
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
MVP may rebuild whole TSDF after keyframe changes; do not implement advanced weighted voxel logic yet.

## Task sequence

- [ ] **M1 — Object map definition.** Immutable object frame and clear map interface; object pose independent from robot base.

- [ ] **M2 — TSDF minimal implementation.** Open3D volume in object coordinates: pass extrinsic T_cam_obj, **not inverse**; valid-mask depth integration.

- [ ] **M3 — Conservative commit.** Only validated and version-consistent keyframes update map; provisional vs committed frames.

- [ ] **M4 — Geometry output.** Object point cloud, mesh, per-frame fused count, debugging overlays, model export transform.

- [ ] **M5 — Baseline tests.** R-STATIC model matches the V1 base-frame model up to the known gauge translation; R-BOTH merges object surfaces without smearing; wrong-extrinsic (inverted) regression test on a real session.

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
