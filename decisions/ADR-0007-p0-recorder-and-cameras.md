# ADR-0007: New recordings only, recorder in P0, head camera recorded
Status: Accepted (owner decisions 2026-10-27). Date: 2026-10-27.

## Owner decisions
1. The hand-eye calibration of 2026-10-12 is the reference from now on: `ref/MVStreamRefine/data/handeye/handeye_tf.txt`, sha256 `db07f1dd5cbc204c8a8d8f91c9d4025572d59fe4d54dcce5916f8d1a2d5fd61f` (block `wrist_cam2 = T_tcp_cam`; block `base_cam1 = T_base_head`). Older calibrations are not used.
2. No V1 data is reused (neither `data/` nor `output/captures/*`). There is no V1 importer.
3. All test and gate data, starting with P0, are new sessions recorded by the owner with the V2 recorder.
4. Code is implemented first; the owner then runs the recorder on the rig and the P0 gate is evaluated on those sessions.
5. The fixed head camera is still in use.

## Decisions
- **P0 includes the recorder.** P0 delivers `mvsr record`: D405 wrist + optional D435 head + RM65 state, writing `contracts/SESSION_FORMAT.md` with every raw timestamp and every robot sample. P1 keeps calibration verification, clock synchronization, interpolation, replay-to-`SensorFrame` and validation analysis.
- **Recorder does not move the robot.** The operator moves the arm (teach pendant / drag teaching / a separate motion tool). Robot motion commands stay outside the tracking/recording core (P9 RT5).
- **Head camera** (D435, serial `244622070961`, fixed, `T_base_head` from `base_cam1`, depth unit `1e-3 m`, color `rgb8` required at 1280×720) is a first-class optional stream in the session format (`camera_id = head`). V2 algorithms remain wrist-only for the first release; using the head stream in tracking/mapping/GT needs a later ADR. The head calibration was not re-done on 2026-10-12 and is unverified.
- **Disk budget.** Workstation system disk has ≈ 113 GB free (2026-10-27). Lossless 1280×720 @ 30 fps is ≈ 6 GB/min per camera. Defaults: wrist 30 fps; head disabled unless requested, and when enabled 15 fps. The recorder refuses to start when free space < configured reserve + estimated session size, and stops cleanly (manifest finalized) when the reserve is reached.
- The two RealSense devices are not hardware-synchronized; each stream keeps its own timestamps and frame ids.

## Consequences
- P0 now has hardware-bound code (`mvsr.devices`, `mvsr.record`). Core/IO modules stay pure CPU; hardware tests are `@pytest.mark.hardware` and run in the `rig` compose service.
- `pyrealsense2` and `Robotic_Arm==1.1.6` become P0 dependencies (image `rig` target).
- P0 gate is BLOCKED until the owner records the gate sessions.
