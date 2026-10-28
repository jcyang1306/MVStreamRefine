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
  - `RM65StateSource`: connect, read joints + TCP `xyzrpy`; polling thread (stamp = request midpoint) with optional UDP real-time push when supported by controller/SDK (verify and record which); transient `rm_get_current_arm_state` ret=-1/-2 retried before a failed sample is logged; `xyzrpy → R` conversion (`Rz·Ry·Rx`) in this adapter only.
  - Port V1 logic through adapters with notes on what changed; no imports from V1.

- [x] **F5 — Recorder (`mvsr record`).** One capture thread per camera → bounded queue → PNG writer pool (lossless, fast compression); robot thread → `robot.jsonl`; every frame gets a `frames.jsonl` record including drops with reason; CLI options: `--cameras wrist[,head]` (default both), per-camera fps, optional `--duration` (omit: operator stops with Ctrl+C / Q), `--scenario` (default `R-STATIC`), `--object`, `--notes`; start/stop via keys or Ctrl+C with clean manifest finalization; optional preview window with live stats (fps, queue depth, drops, robot rate, free disk); disk budget check before and during recording (ADR-0007). `mvsr inspect-session` prints measured duration and frame counts from the records; `mvsr verify-session [--p0-gate]` checks integrity and P0 completeness.

- [x] **F6 — Tests.** Analytic + SDK-double tests in `tools/check.sh`. Hardware pytest (`tests/hardware`, 10 s auto-stop for CI/rig smoke) passed on CW20175 2026-10-27 (wrist-only and wrist+head).
  - Analytic: Exp/Log round-trip incl. θ→0 and θ→π, numerical Jacobians, A/B/C compose order, inverse, frame-mismatch errors, covariance adjoint, quaternion sign canonicalization, interpolation endpoints/midpoints, `xyzrpy` conversion vs. scipy, depth unit conversion, JSON/NPZ round-trip, schema-version rejection, atomic-write interruption, checksum mismatch, hand-eye parser on the reference file.
  - Recorder control flow with SDK test doubles (ADR-0002: error paths only): device disconnect, wrong depth unit, robot read failures (transient ret=-2 recovered by retry; exhausted retries logged), full queue, disk reserve reached, Ctrl+C finalization.
  - `@pytest.mark.hardware` smoke tests on the rig: 10 s wrist-only and 10 s wrist+head recordings verify cleanly (`--duration` is only for this automated pytest, not the operator P0 take).

## Gate (I-P0) — one operator-stopped recording

One take is enough. Do not pre-set a capture length; stop with Ctrl+C or Q when the viewpoints are enough. `inspect-session` / `verify-session` count frames and duration from the records.

```bash
xhost +si:localuser:$USER
docker compose run --rm rig mvsr record --object <name> --gt constant_T_base_obj
# default --cameras wrist,head --scenario R-STATIC, live preview on; Q stops
docker compose run --rm dev mvsr inspect-session <session_id>
docker compose run --rm dev mvsr verify-session <session_id> --p0-gate
docker compose run --rm dev mvsr register-session <session_id> --used-by P0-gate --gt constant_T_base_obj
```

**PASS** (all of these, and nothing else):

1. Manifest `status=COMPLETE` (clean operator stop, not ABORTED).
2. `verify-session` integrity: checksums, contiguous `frame_id`, every drop logged with a reason, reader round-trip `depth_m = raw * depth_units_m`.
3. `--p0-gate`: `wrist.frames_written > 0`, `robot.samples > 0`, and if head was in the session then `head.frames_written > 0`.
4. Raw timestamp fields populated or explicitly `null` per device (reported, not a numeric target).
5. Session registered in `fixtures/registry.yaml`.

**Not gated (reported only):** wrist/head fps, USB dual-cam contention, robot poll Hz / jitter, leftover robot read failures after retries. Dual-cam USB sharing and poll jitter are assumed not to set later model accuracy for this gate. R-CALIB remains optional and feeds P1, not I-P0.

`--duration` is optional auto-stop for hardware pytest / convenience only. The historical SMOKE (wrist dress rehearsal) and timed R-STATIC takes are not required going forward.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run I-P0 in `../INTEGRATION_PLAN.md` on owner-recorded sessions.
- Performance log: per-camera capture fps, writer throughput, queue high-water mark, CPU load, disk MB/s, robot sample rate and jitter.

## Completion checklist
- [x] API documentation and schema tests complete.
- [x] Deterministic offline test passes and reproduction command recorded (`docker compose run --rm dev tools/check.sh`).
- [x] Integration gate evidence and quantitative metrics saved. Registered P0-gate sessions: `20261027T103359_CW20175_9605` (wrist SMOKE), `20261027T105934_CW20175_f4d5` (wrist+head R-STATIC). Live `tests/hardware` passed on CW20175. Going forward a single operator-stopped take + `verify-session --p0-gate` is sufficient.
- [x] Known limitations recorded; no false success or hidden fallback.
- [x] Handoff report names changed files, tests, residual risks and next phase (below).

## Handoff (P0 → P1)

**What a gate record is:** the machine-checkable evidence for I-P0: session directory + `manifest.json` checksum, `mvsr verify-session --p0-gate` report (errors empty), `mvsr inspect-session` measured duration/frame counts/rates, and the `fixtures/registry.yaml` entry (`used_by: [P0-gate]`). It is not a narrative.

**What a handoff is:** this section — what shipped, how to reproduce, what is still risky, what the next phase must do. It is for the next implementer, not a second recording.

**Shipped:** `src/mvsr/{core,io,calib,devices,record,cli.py}`, `configs/record.yaml`, `configs/handeye/handeye_tf_20261012.txt`, `tests/unit`, `tests/hardware`, `tests/test_registered_sessions.py`, Docker `dev`/`rig`. Recorder never commands the robot.

**Reproduce:** `docker compose run --rm dev tools/check.sh`. Operator P0 take: `mvsr record --object <name> --gt constant_T_base_obj` then `verify-session --p0-gate`. Rig smoke: `docker compose run --rm rig pytest -q tests/hardware --run-hardware`.

**Residual risks:** D435 `exposure_us` often null; dual-cam USB can lower delivered fps (not gated); RM65 poll can still log `ret=-2` after retries (session continues unless failures last `abort_after_failure_s`); UDP push unverified; R-CALIB not recorded (P1).

**Next:** P1 `subplans/01_SENSORS.md` / I0 — replay with clock mapping and interpolated robot poses; R-CALIB recommended.

## Status (2026-10-28)
F0–F6 and I-P0 complete under the single-take PASS definition. Hardware pytest and two owner sessions already exist; fps/Hz are diagnostics only. RM65 transient pose reads retry up to 3 times.
