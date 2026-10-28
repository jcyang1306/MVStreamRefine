# P1 — Sensors

**Dependencies:** P0
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Output only SensorFrame; no SAM, ICP, keyframes or object pose. Check the static board inside the motion this rig can actually reach. `dt_robot_cam_ns` stays 0 (ADR-0004).

## Task sequence

Device adapters and the recorder were moved to P0 (ADR-0007). P1 turns recorded sessions into trustworthy `SensorFrame`s. Code: `src/mvsr/sensors/`, settings `configs/sensors.yaml`.

- [x] **S1 — Live source.** `LiveSource` (`sensors/live.py`) wraps `RealSenseStream` + `RM65StateSource` (read-only) and emits `SensorFrame` online; after acquisition it uses the same `build_sensor_frame`, clock function and RGB-D pairing rule as replay. Rig test `tests/hardware/test_live_source.py`: 90/90 frames INTERPOLATED, arrival − mid-exposure 27.8–28.3 ms.

- [x] **S2 — Calibration verification.** `sensors/calib_check.py` + `sensors/board.py`: 11×8 board (SB detector, orientation canonicalized by handedness + first-square colour), RealSense `inverse_brown_conrady` undistortion, IPPE + LM PnP. Board-in-base spread, hand-eye reprojection residual, depth bias vs. distance, planarity and square pitch; head `T_base_head` against the wrist reference. Results go into new `CalibrationBundle`s (notes appended) under the run directory; raw session bundles are not modified.

- [x] **S3 — Recording protocol.** Checklist below; one R-CALIB take (`20261028T083612_CW20175_5bf8`) plus the P0 R-STATIC take (`20261028T080558_CW20175_9fb3`) registered `used_by: I0`.

- [x] **S4 — Clock synchronization.** `sensors/clock.py`: mid-exposure from RealSense global time (primary, ADR-0004 §3 amended), exact lower-envelope fit as cross-check/fallback with residuals, uint32 µs counter unwrap; `sensors/sync.py`: bracketing slerp+lerp with the 25 ms gap limit, no extrapolation, `PoseSyncInfo` per frame. `dt_robot_cam` not estimated (applied value 0 from the bundle).

- [x] **S5 — Replay.** `ReplaySource` (`sensors/replay.py`, port `SensorSource.next() -> SensorFrame | EndOfStream`): checksum-verified, deterministic (`mvsr replay` prints an output sha256; identical across runs), `speed=None|s`, random-access `read(frame_id)`.

- [x] **S6 — Validation sessions.** `mvsr i0-gate` on the two registered takes; evidence below.

### Found during P1 (not in the original plan)
- **Colour/depth from different exposures** (ADR-0004 §8): 14–15 % of wrist framesets (repeated colour frame #N with new depth, or depth one interval off) and the same on the head. Sources do not emit them and count them per reason. Owner decision 2026-10-28: keep this behaviour; a wrist-only comparison recording is deferred.
- **Head extrinsic rechecked on the existing R-CALIB take** (`mvsr sensors-report --handeye configs/handeye/handeye_tf_20261028.txt`, run `data/runs/20261028T095657_CW20175_0937`; the session copy stays 2026-10-12). Head board vs wrist board: 31.0 mm / 4.17° with the old extrinsic, **2.51 mm / 0.60°** with the new one. The new head pose also matches `targettobase` in the calibration JSON to 0.32 mm / 0.04°. Wrist board spread p95 moved from 3.29 mm / 0.94° to 3.83 mm / 0.79°, and hand-eye reprojection p95 from 4.77 px to 6.05 px; both still inside the placeholder gate. The JSON quality flag remains `exploratory`.

## SensorFrame `source_meta` keys (both sources)
`source` (`replay|live`), `session_id`, `not_written_before` (records without pixels since the previous emitted frame), `rgbd_rejected_before`, `rgbd` (`OK|UNCHECKED|REPEATED_COLOR|COLOR_DEPTH_MISMATCH`), `color_depth_dt_ms`.

## Recording protocol (S3)
All scenarios: `docker compose run --rm rig mvsr record --object <name> --scenario <TAG> [--gt <type>] --notes "..."`, preview on, stop with Q; then `inspect-session`, `verify-session --p0-gate`, `register-session --used-by <gate>`.
- **Lighting:** diffuse room light, no direct sun or specular hot spots on the board/object; auto-exposure on (recorded in the manifest).
- **Distance:** D405 7–50 cm; keep the target at 25–45 cm (R-CALIB board measured at 33–41 cm).
- **Robot motion:** operator-driven, the reachable viewpoints of this cell; one take, no speed ladder (ADR-0004 §7). Avoid stops longer than a few seconds with nothing changing.
- **Disk:** recorder refuses to start below the 20 GB reserve; dual camera ≈ 157 MB/s.
- **R-CALIB:** 11×8 board fixed on the table, all 88 inner corners visible in the wrist view, `--gt constant_board_in_base --notes "square_mm=<caliper value>"`.
- **R-STATIC:** object fixed on the table, `--gt constant_T_base_obj`.
- **Annotations:** A in the preview marks a moment (events.jsonl); describe what it means in `--notes`.

## Gate evidence (I0, 2026-10-28)
Run `data/runs/20261028T091606_CW20175_17a7` (`sensors_report.json`, verified bundles, manifest with sha256); reproduce with `docker compose run --rm dev mvsr i0-gate`. Decision **PASS** against the placeholder targets in `configs/sensors.yaml: gate`.

| Metric | R-CALIB `…5bf8` | R-STATIC `…9fb3` |
|---|---|---|
| wrist frames emitted / written | 1021 / 1188 | 1090 / 1288 |
| wrist RGB-D rejected (repeat / mismatch) | 51 / 116 | 79 / 119 |
| wrist clock drift, transfer latency | −68.0 ppm, 28.0 ms | −67.3 ppm, 27.3 ms |
| wrist clock residual p95 (envelope vs global) | 0.096 ms | 0.081 ms |
| wrist robot prior INTERPOLATED | 98.2 % | 99.4 % |
| board-in-base spread median / p95 | 1.61 / 3.29 mm, 0.38 / 0.94° | — |
| nearest-sample prior spread p95 | 3.29 mm | — |
| hand-eye reprojection median / p95 | 2.10 / 4.77 px | — |
| PnP RMS p95 | 0.36 px | — |
| depth bias median (33–41 cm) | +0.77 mm | — |
| depth \|plane distance\| median / p90 | 2.2 / 6.5 mm | — |
| square pitch from depth / nominal 15 mm | 0.9989 | — |
| head vs wrist board reference | 31.0 mm / 4.17° | — |

Interpolated and nearest-sample priors give the same spread here, so robot pose noise is not limited by pairing at this motion speed.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run the matching I-gate in `../INTEGRATION_PLAN.md` by replaying registered real sessions (`fixtures/registry.yaml`); live-hardware checks are additional, marked `hardware`.
- Performance/accuracy log: include relevant latency, failure cases, quality and version numbers.

## Completion checklist
- [x] API documentation and schema tests complete (module docstrings; `test_clock`, `test_sync`, `test_board`, `test_rgbd`, `test_replay`, `test_live`, `test_sensors_config`).
- [x] Deterministic offline test passes and reproduction command recorded (`docker compose run --rm dev tools/check.sh`: 171 passed, 3 hardware skipped; `tests/test_i0_sessions.py` on the registered takes).
- [x] Integration gate evidence and quantitative metrics saved (run above).
- [x] Known limitations recorded; no false success or hidden fallback.
- [x] Handoff report names changed files, tests, residual risks and next phase (below).

## Handoff (P1 → P2)
**Shipped:** `src/mvsr/sensors/{config,clock,sync,rgbd,frame,source,replay,live,board,calib_check,report,stats}.py`, `src/mvsr/yaml_config.py` (shared typed YAML, used by `record/config.py`), `configs/sensors.yaml`, CLI `mvsr replay | sensors-report | i0-gate`, tests listed above, `tests/test_i0_sessions.py`, `tests/hardware/test_live_source.py`.

**Reproduce:** `docker compose run --rm dev mvsr i0-gate`; `docker compose run --rm dev mvsr replay <session_id>` (sha256 must not change between runs); rig: `docker compose run --rm rig pytest -q tests/hardware --run-hardware`.

**Owner decisions (2026-10-28):** timestamp amendment in ADR-0004 accepted; colour/depth mismatch handling stays as implemented and is not retested now; the 2026-10-28 hand-eye file is the reference for new recordings.

**Residual risks:** `dt_robot_cam` not estimated (robot poll midpoint vs. controller sample time untested); all gate targets are placeholders; spread/residual hold only inside this cell's motion envelope; 15 % of wrist frames lost to colour/depth mismatch on dual-camera takes (accepted for now); the new hand-eye is marked exploratory even though this R-CALIB take agrees with it; D405 raw depth at 35–40 cm has a 6.5 mm p90 plane deviation (device filters off — P2 must filter). New recordings copy `handeye_tf_20261028.txt`; the two registered sessions still contain the 2026-10-12 copy.

**Next:** P2 `subplans/02_OBSERVATION.md` consumes `ReplaySource` frames (`T_base_cam_prior`, `depth_m_f32`, `valid_depth_mask`) and must handle `T_base_cam_prior=None` frames.
