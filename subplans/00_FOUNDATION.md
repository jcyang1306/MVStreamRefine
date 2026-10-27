# P0 — Core, Session IO & Recorder

**Dependencies:** none
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`, `../contracts/SESSION_FORMAT.md`, ADR-0002…0007.

## Purpose and boundary
Foundation plus the data recorder (ADR-0007): transform math, contract types, session IO, calibration loading, device adapters and `mvsr record`. Core and IO stay pure CPU; device/recorder code is hardware-bound and isolated in `mvsr.devices` / `mvsr.record`. No SAM, tracking, clock-model fitting or pose interpolation at replay (those are P1+). The recorder never commands robot motion. No synthetic datasets; no V1 data.

## Task sequence

- [x] **F0 — Bootstrap repository.** `git init` (ignore `ref/`, `data/`, outputs); `pyproject.toml` (`src/mvsr`, Python 3.10, ruff, mypy, pytest, hypothesis); `Dockerfile` with `dev` and `rig` targets and `compose.yaml` services (ADR-0006); `mvsr` CLI skeleton (`--version`, `record`, `inspect-session`, `verify-session`); `fixtures/registry.yaml`; `$MVSR_DATA_ROOT` (default `./data`).

- [x] **F1 — Coordinate math (`mvsr.core.se3`).** `PoseSE3` with frame names and covariance; so3/se3 `Exp`/`Log` (exact, small-angle safe), hat/vee, adjoint, left Jacobian; compose/inverse with frame checks and covariance propagation; `transform_points`; quaternion `xyzw` ↔ R (canonical `w ≥ 0`); decoupled (slerp + lerp) and SE(3)-geodesic interpolation as separately named functions.

- [x] **F2 — Contract types (`mvsr.core.types`, `mvsr.core.errors`).** Frozen dataclasses and enums exactly as in `DATA_CONTRACTS.md`; validators; non-writeable arrays; JSON/NPZ (de)serialization with `schema_version`; deterministic `calibration_id`; errors (`FrameMismatchError`, `ChecksumMismatchError`, `SchemaVersionError`, `InvalidPoseError`, `DiskBudgetError`).

- [x] **F3 — Session IO and calibration (`mvsr.io`, `mvsr.calib`).** Atomic writer (tmp → fsync → rename), sha256 utilities; `SessionWriter` (streaming, multi-camera, manifest lifecycle `RECORDING → COMPLETE | ABORTED`); `SessionReader` (frames, robot samples, events; one-time depth conversion; checksum verification); hand-eye parser (`wrist_cam2 = T_tcp_cam`, `base_cam1 = T_base_head`) for the reference file (ADR-0007, sha256 checked); `CalibrationBundle` built from SDK intrinsics at record time + hand-eye.

- [x] **F4 — Device adapters (`mvsr.devices`).**
  - `RealSenseStream(camera_id, serial, width, height, fps, color_format)`: color + z16 depth aligned to color; global time enabled; device post-filters off; reads per-frame `get_timestamp()` + domain, `SENSOR_TIMESTAMP`, `FRAME_TIMESTAMP`, `ACTUAL_EXPOSURE`, frame numbers (each `None` when unsupported, never invented); verifies serial and `depth_units_m` (D405 `1e-4`, D435 `1e-3`); head D435 uses `rgb8`.
  - `RM65StateSource`: connect, read joints + TCP `xyzrpy`; polling thread (stamp = request midpoint) with optional UDP real-time push when supported by controller/SDK (verify and record which); `xyzrpy → R` conversion (`Rz·Ry·Rx`) in this adapter only.
  - Port V1 logic through adapters with notes on what changed; no imports from V1.

- [x] **F5 — Recorder (`mvsr record`).** One capture thread per camera → bounded queue → PNG writer pool (lossless, fast compression); robot thread → `robot.jsonl`; every frame gets a `frames.jsonl` record including drops with reason; CLI options: `--cameras wrist[,head]`, per-camera fps (head default 15), `--duration`, `--scenario`, `--object`, `--notes`; start/stop via keys or Ctrl+C with clean manifest finalization; optional preview window with live stats (fps, queue depth, drops, robot rate, free disk); disk budget check before and during recording (ADR-0007). `mvsr inspect-session` prints a summary report; `mvsr verify-session` checks integrity.

- [ ] **F6 — Tests.** (analytic + SDK-double tests done; hardware smoke written, not yet run on the rig)
  - Analytic: Exp/Log round-trip incl. θ→0 and θ→π, numerical Jacobians, A/B/C compose order, inverse, frame-mismatch errors, covariance adjoint, quaternion sign canonicalization, interpolation endpoints/midpoints, `xyzrpy` conversion vs. scipy, depth unit conversion, JSON/NPZ round-trip, schema-version rejection, atomic-write interruption, checksum mismatch, hand-eye parser on the reference file.
  - Recorder control flow with SDK test doubles (ADR-0002: error paths only): device disconnect, wrong depth unit, robot read failures, full queue, disk reserve reached, Ctrl+C finalization.
  - `@pytest.mark.hardware` smoke tests on the rig: 10 s wrist-only and 10 s wrist+head recordings verify cleanly.

## Gate (I-P0) — needs owner-recorded sessions
Code is complete first; the owner then records on the rig:
1. **SMOKE**: 60 s, wrist only, object static, robot moved by the operator.
2. **R-STATIC**: 2 min, wrist 30 fps + head 15 fps, object static, robot moved through varied viewpoints.
3. (Recommended, feeds P1) **R-CALIB**: 11×8 checkerboard static, robot moved slow / medium / fast.

PASS when, for every session: `verify-session` succeeds (all checksums, contiguous `frame_id`, every drop logged with a reason, manifest `COMPLETE`); wrist written ratio and robot sample rate/jitter are reported, with targets wrist ≥ 99 % written at 30 fps and robot ≥ 50 Hz (placeholders, adjusted after the first measurement); recorded raw timestamp fields are populated or explicitly `null` per device capability; reader round-trip reproduces raw depth × `depth_units_m` exactly; sessions are registered in `fixtures/registry.yaml`.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run I-P0 in `../INTEGRATION_PLAN.md` on owner-recorded sessions.
- Performance log: per-camera capture fps, writer throughput, queue high-water mark, CPU load, disk MB/s, robot sample rate and jitter.

## Completion checklist
- [x] API documentation and schema tests complete.
- [x] Deterministic offline test passes and reproduction command recorded (`docker compose run --rm dev tools/check.sh`).
- [ ] Integration gate evidence and quantitative metrics saved. **BLOCKED**: needs owner-recorded SMOKE / R-STATIC (/ R-CALIB) sessions and the live `tests/hardware` run.
- [x] Known limitations recorded; no false success or hidden fallback.
- [ ] Handoff report names changed files, tests, residual risks and next phase (draft in the P0 implementation chat; finalize after the gate).

## Status (2026-10-27)
F0–F6 code complete; `tools/check.sh`: ruff/format clean, `mypy --strict mvsr.core` clean,
126 passed / 3 skipped (2 hardware, 1 registry-empty). Not yet run against the real D405/D435/RM65
(devices not attached at implementation time). Open items for the rig: verify `actual_exposure` units,
capture fps with CPU alignment at 1280×720, RM65 poll rate/latency, UDP push (`--robot-source udp`,
unverified), then tune the PLACEHOLDER values in `configs/record.yaml`.
