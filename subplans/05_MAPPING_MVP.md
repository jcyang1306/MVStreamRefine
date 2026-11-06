# P5 — Object-centric Mapping MVP

**Dependencies:** P3 + P4
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
MVP may rebuild whole TSDF after keyframe changes; do not implement advanced weighted voxel logic yet. Historical pose-update reintegration remains P7: `apply_pose_updates` rejects rather than moving an already integrated volume.

## Task sequence

- [x] **M1 — Object map definition.** `mapping/map.py` `ObjectMap` owns one immutable `T_base_obj_init` and the contract ports `commit`, `view`, `snapshot`. Object geometry is stored in `obj`, independent of the robot base.

- [x] **M2 — TSDF minimal implementation.** `mapping/open3d_backend.py` integrates masked depth with Open3D `ScalableTSDFVolume`. The only extrinsic passed is `T_cam_obj`; depth outside `geometry_mask` is invalid and is never filled. Voxel size is a representation setting, not an accuracy claim.

- [x] **M3 — Conservative commit.** Commit requires `PoseStatus.VALID`, matching keyframe pose/version, minimum geometry, and the session tracking-spread gate. Repeating the same keyframe/version is idempotent; a stale version is rejected. `build_provisional` writes a separate diagnostic volume and never increments `map_version`.

- [x] **M4 — Geometry output.** Point cloud, mesh, bounds and fused counts are reported. The model is the reachable surface only (ADR-0008). Provisional files live under `provisional/`; a committed `MapSnapshot` is published only after I3 PASS.

- [x] **M5 — Baseline tests.** Held-out keyframe observations, a static robot-prior fusion, and an intentionally inverted extrinsic are compared. R-BOTH smearing remains untested until that session exists.

## Gate evidence (I3, 2026-11-06)
Session `20261028T080558_CW20175_9fb3`, P4 memory from the latest compatible run, 52 fused / 12 held-out keyframes. Run `data/runs/20261106T094730_CW20175_180d`. Reproduce: `docker compose run --rm dev mvsr i3-gate`.

**Decision: BLOCKED.** The mapper itself works, but committed fusion is refused:

| Check | Result |
|---|---|
| provisional object points / mesh | 6071 points, 6193 vertices, 10421 triangles; repeat digest identical |
| held-out self-consistency | median 2.72 mm, p95 8.93 mm, coverage 0.965 |
| static robot-prior baseline | median 2.95 mm, p95 9.97 mm, coverage 0.950 |
| inverted extrinsic | p95 96.1 mm, error ratio 10.8× |
| selected-keyframe `T_base_obj` p95 | 195.5 mm / 30.0° (gate 15 mm / 3°) |
| R-BOTH | absent |

Held-out residual uses the same estimated poses on both the model and the query, so it measures local fusion consistency, not independent shape accuracy. The robot-prior baseline is the meaningful static comparison and is similar. No committed snapshot was written.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run the matching I-gate in `../INTEGRATION_PLAN.md` by replaying registered real sessions (`fixtures/registry.yaml`); live-hardware checks are additional, marked `hardware`.
- Performance/accuracy log: include relevant latency, failure cases, quality and version numbers.

## Completion checklist
- [x] API documentation and schema tests complete.
- [x] Deterministic offline test passes and reproduction command recorded.
- [x] Integration gate evidence and quantitative metrics saved (I3 BLOCKED; provisional model saved).
- [x] Known limitations recorded; no false success or hidden fallback.
- [x] Handoff report names changed files, tests, residual risks and next phase.

## Handoff (P5 → P6/P7)
**Shipped:** `src/mvsr/mapping/{config,open3d_backend,map,benchmark,report}.py`, `configs/mapping.yaml`, Open3D pin, CLI `mvsr mapping-report | i3-gate`.

**Residual risks:** committed mapping stays disabled while keyframe `T_base_obj` p95 is ~196 mm / 30°; R-BOTH is missing, so smearing is untested; held-out self-consistency must not be quoted as shape GT. P6 must reduce keyframe-pose inconsistency before I3 can pass. P7 owns rebuild after accepted pose updates.
