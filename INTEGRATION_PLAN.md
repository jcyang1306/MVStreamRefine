# Integration Plan — incremental end-to-end gates

## Rules
No integration by copying private state across modules. Use `contracts/DATA_CONTRACTS.md`. Before each integration, run producer and consumer contract tests; replay registered real sessions (`fixtures/registry.yaml`) and save a reproducible report. No synthetic datasets (ADR-0002).

## Recording scenarios (real rig)
Scenario codes are defined in ADR-0002. Each gate lists the minimum sessions it needs; all sessions are recorded with the V2 recorder `mvsr record` (P0, ADR-0007).

| Code | Setup | GT available |
|---|---|---|
| R-CALIB | static 11×8 checkerboard, one take at the reachable viewpoints and speed | board pose in `base` over that motion; hand-eye residual inside the envelope; board planarity / square pitch. No `dt_robot_cam` fit (ADR-0004) |
| R-STATIC | object fixed on table, robot moving | `T_base_obj` constant |
| R-MARKER | object rigidly on one plate with the 11×8 checkerboard beside it (or a ChArUco board), plate moved by hand, robot moving | independent `T_cam_obj` up to one constant |
| R-HANDHELD | robot static, object moved by hand | held-out surface residual only |
| R-BOTH | robot and object moving | held-out surface residual; marker if attached |
| R-HARD | fast motion, hand occlusion, out-of-view + re-entry, >90° rotation | loss/recovery events annotated in `events.jsonl` |
| Object set | textured small object, low-texture geometric object, symmetric object, reflective object | shape GT object for at least one |

## Gates
- **I-P0 (core + recorder):** one operator-stopped `mvsr record` (default wrist+head, `R-STATIC`); `verify-session --p0-gate` passes (COMPLETE, wrist/robot frames present, head frames if recorded, drops logged, depth round-trip exact). Duration and fps are measured from the session, not pre-set. R-CALIB is recommended for I0, not required here. Details in `subplans/00_FOUNDATION.md`.
- **I0 (core + sensors):** replay with device-to-host clock mapping and robot interpolation at `dt_robot_cam_ns = 0`. One R-CALIB take: no unlogged drops, camera timestamp mapping residuals, board-in-base spread and hand-eye residual inside the recorded envelope. Speed sweeps and a fitted time offset are out of scope. Run `mvsr i0-gate` (registry sessions `used_by: I0`). BLOCKED if a hard check fails (no R-CALIB, unlogged frames, wrist prior < 95 %, < 100 board poses); PASS-WITH-LIMITS if a soft target in `configs/sensors.yaml: gate` is missed (clock residual p95 ≤ 1 ms, board spread p95 ≤ 5 mm / 1°, hand-eye reprojection p95 ≤ 10 px, RGB-D valid ≥ 80 %; all placeholders). **2026-10-28: PASS**, evidence in `subplans/01_SENSORS.md`.
- **I1 (sensors + observation):** same frame → reproducible mask and keypoint/track output for a fixed `mask_revision`; R-HARD clips for invalid depth, occlusion, object leaving frame.
- **I2 (+ tracking):** R-STATIC: `T_base_obj(t)` spread (median/P95 translation, rotation); R-MARKER: RPE/ATE vs. board after fixed-gauge alignment; R-BOTH: failure %, degeneracy flags; output `T_cam_obj` and `T_base_obj`.
- **I3 (+ keyframes + MVP mapping):** object-frame TSDF; R-STATIC result matches V1 base-frame model up to the known gauge translation; R-BOTH shows no smearing; held-out residual vs. robot-pose-only baseline.
- **I4 (+ local BA):** keyframe pose updates; R-MARKER RPE and held-out surface error improve or regression is documented; low-confidence frames not fused.
- **I5 (+ reintegration):** apply pose updates to committed keyframes and rebuild from immutable observations; result identical to a clean integration at the corrected poses.
- **I6 (+ relocalization):** R-HARD occlusion / out-of-view / symmetric object; same object map reused on verified recovery, gauge unchanged.
- **I7 (+ realtime):** live run on the rig: bounded queues, no stale-pose fusion, throughput, clean stop/resume/restart, watchdogs, latency trace; the recorded copy of the live run replays to the same committed map.
- **I8 (robot pose correction, optional):** fixed camera prior vs. regularized correction on R-MARKER held-out GT; retain only with measurable improvement.

## Deliverables for each gate
- baseline and current metrics (median, RMSE, P95, failure %, runtime and memory)
- known limitations and failure cases
- command to reproduce, config hash, git hash, session ids + manifest checksums, calibration_id, mask_revision
- decision: PASS / BLOCKED / PASS-WITH-LIMITS and explicit follow-up issue
