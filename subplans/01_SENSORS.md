# P1 — Sensors

**Dependencies:** P0
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Output only SensorFrame; no SAM, ICP, keyframes or object pose. Benchmark static target misalignment versus speed and timing.

## Task sequence

Device adapters and the recorder were moved to P0 (ADR-0007). P1 turns recorded sessions into trustworthy `SensorFrame`s.

- [ ] **S1 — Live source.** `LiveSource` wrapping the P0 device adapters and emitting `SensorFrame` online (used by P9); same code path as replay after acquisition.

- [ ] **S2 — Calibration verification.** Hand-eye residual on R-CALIB (board-in-base spread); D405 depth bias vs. distance and scale from board planarity and measured square pitch; record results in `CalibrationBundle.notes` and a report. Head `T_base_head` checked the same way when head data exists.

- [ ] **S3 — Recording protocol.** Written checklist per scenario (lighting, distance range 7–50 cm for D405, robot speeds, operator annotations, disk budget) and first R-CALIB / R-STATIC recordings beyond the P0 gate sessions.

- [ ] **S4 — Clock synchronization.** Device→host time fit with residuals; mid-exposure estimate; robot interpolation with gap limits (no extrapolation); `PoseSyncInfo` per frame; `dt_robot_cam` estimation from R-CALIB.

- [ ] **S5 — Replay.** `ReplaySource`: deterministic replay of recorded sessions to `SensorFrame` (clock mapping + interpolated `T_base_cam_prior` + `PoseSyncInfo`) at any speed; identical output across runs.

- [ ] **S6 — Validation sessions.** Record R-CALIB (3 robot speeds, static board) and a first R-STATIC session; report board-in-base spread vs. speed with nearest-vs-interpolated pairing and with/without `dt_robot_cam`, hand-eye reprojection residual, robot sample-rate/jitter and drop histograms.

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
