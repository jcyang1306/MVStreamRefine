# Integration Plan — incremental end-to-end gates

## Rules
No integration by copying private state across modules. Use `contracts/DATA_CONTRACTS.md`. Before each integration, run producer and consumer contract tests; replay registered real sessions (`fixtures/registry.yaml`) and save a reproducible report. No synthetic datasets (ADR-0002).

## Recording scenarios (real rig)
Scenario codes are defined in ADR-0002. Each gate lists the minimum sessions it needs; all sessions are recorded with the V2 recorder `mvsr record` (P0, ADR-0007).

| Code | Setup | GT available |
|---|---|---|
| R-CALIB | static 11×8 checkerboard, robot moving at 3 speeds | board pose in `base` must be constant; time offset; hand-eye residual; board planarity / square pitch for depth bias and scale |
| R-STATIC | object fixed on table, robot moving | `T_base_obj` constant |
| R-MARKER | object rigidly on one plate with the 11×8 checkerboard beside it (or a ChArUco board), plate moved by hand, robot moving | independent `T_cam_obj` up to one constant |
| R-HANDHELD | robot static, object moved by hand | held-out surface residual only |
| R-BOTH | robot and object moving | held-out surface residual; marker if attached |
| R-HARD | fast motion, hand occlusion, out-of-view + re-entry, >90° rotation | loss/recovery events annotated in `events.jsonl` |
| Object set | textured small object, low-texture geometric object, symmetric object, reflective object | shape GT object for at least one |

## Gates
- **I-P0 (core + recorder):** owner records SMOKE, R-STATIC (wrist + head) and, recommended, R-CALIB with `mvsr record`; `verify-session` passes; drops all logged with reasons; capture/robot rates and raw timestamp field coverage reported; reader round-trip exact (details in `subplans/00_FOUNDATION.md`).
- **I0 (core + sensors):** replay with clock mapping and interpolation; R-CALIB at 3 speeds: no unlogged drops, robot sample rate/jitter histogram, camera timestamp mapping residuals, estimated `dt_robot_cam`, board-in-base spread vs. speed (with/without interpolation and offset).
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
