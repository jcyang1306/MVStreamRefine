# P4 — Keyframe Memory

**Dependencies:** P3
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Keyframes stores measurements, not the only copy of geometry; do not mutate original frames after optimization. Selection is by object-frame viewpoint and covisibility (ADR-0008); no loop-closure keyframe is expected.

## Task sequence

- [x] **K1 — Canonical keyframe.** `keyframes/manager.py` creates contract `Keyframe`s with immutable `ObjectObservation` NPZ references, source RGB/depth/mask paths + sha256, camera prior, provisional/current object poses, track IDs, calibration header, version and canonical checksum.

- [x] **K2 — Selection.** Deterministic selection combines object-frame camera centre/optical-axis novelty, pose overlap, 5 mm visibility-voxel Jaccard, ORB majority-bit appearance, 4×4 feature occupancy, quality, minimum spacing and maximum gap. The first valid P3 gauge frame is always accepted to break the P3/P4 bootstrap cycle.

- [x] **K3 — Local window.** P3 `TrackingSession` now accepts a P4 memory provider. Before each pose estimate P4 ranks global memory against the predicted pose/current observation, greedily diversifies view directions, bounds the local set, and includes the latest anchor. Versioned `KeyframeView`s remain read-only.

- [x] **K4 — Global memory.** Stable monotonically increasing IDs; bounded global memory; appearance, visibility and viewpoint signatures; revisits reuse the nearest existing keyframe instead of creating a loop edge. Capacity rejection is explicit and counted.

- [x] **K5 — Persistence.** A P4 run writes `keyframes/<session>/observations/*.npz`, `keyframes/*.json`, then the memory manifest last. The manifest includes calibration, immutable `T_base_obj_init` and next stable ID; `ObjectTracker.restore()` restores the gauge without reinitializing it. Load verifies run-file hashes, contract checksums, observation/header/revision/track-ID consistency, source session manifest, raw RGB/depth/mask references, mask/config revisions and memory digest. Absolute paths, `..`, backslash ambiguity and symlink escapes are rejected; interrupted directories without a manifest and corrupted files are rejected.

- [x] **K6 — Validation.** `mvsr keyframe-report | p4-gate`: selection decisions/novelty, best-anchor retrieval hit rate, local-window size, revisit reuse, capacity, immutability, persistence reload, deterministic keyframe/pose digests and P3 R-STATIC spread.

## Stage-gate evidence (2026-11-05)
Full session `20261028T080558_CW20175_9fb3` (1090 frames), mask revision `sam2.1-tiny@7558ff4ba813`; run `data/runs/20261105T102446_CW20175_967d`. Reproduce: `docker compose run --rm dev mvsr p4-gate`.

**P4 decision: PASS** (this is not I3; P5 mapping is still absent):

| Metric | Result |
|---|---:|
| selected / capacity-rejected / revisit-reused | 64 / 97 / 637 |
| insertion gap median / p95 / max | 11 / 45.3 / 91 frames |
| best-anchor retrieval hit rate | median/min 1.00 / 1.00 |
| local keyframes | median/max 6 / 6 |
| revisit reuse ratio | 0.9755 |
| deterministic replay | identical choices + memory/pose digests on 120 frames |
| persistence | reload digest identical; raw refs/checksums verified |
| runtime | 3.64 fps |

Tracking with P4 memory improves the P3 full-take p95 from 217.2 mm / 34.1° to **194.3 mm / 29.7°**, but remains far outside the provisional 15 mm / 3° target. P4 therefore passes its memory/retrieval contract while the end-to-end pose quality remains a documented blocker for committed mapping.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run the matching I-gate in `../INTEGRATION_PLAN.md` by replaying registered real sessions (`fixtures/registry.yaml`); live-hardware checks are additional, marked `hardware`.
- Performance/accuracy log: include relevant latency, failure cases, quality and version numbers.

## Completion checklist
- [x] API documentation and schema tests complete.
- [x] Deterministic offline test passes and reproduction command recorded.
- [x] Stage-gate evidence and quantitative metrics saved.
- [x] Known limitations recorded; no false success or hidden fallback.
- [x] Handoff report names changed files, tests, residual risks and next phase.

## Handoff (P4 → P5/P6)
**Shipped:** `src/mvsr/keyframes/{config,manager,benchmark,report}.py`, `configs/keyframes.yaml`, P3 provider integration, CLI `mvsr keyframe-report | p4-gate`, persistence/restart tests and `tests/test_p4_sessions.py`.

**Residual risks:** the 64-keyframe cap is reached on this single take; visibility signatures depend on provisional P3 poses; best-anchor recall is a deterministic internal proxy rather than independent GT; only the low-texture cup/R-STATIC session exists; tracking drift remains too high for committed TSDF.

**Next:** do not interpret P4 PASS as I3. P5 may implement an explicitly provisional/non-committed mapping baseline for diagnostics, but committed mapping should wait for R-MARKER/R-BOTH evidence and P6 keyframe-pose optimization.
