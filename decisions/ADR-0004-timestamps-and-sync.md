# ADR-0004: Timestamps, clock domains and robot pose synchronization
Status: Accepted. Date: 2026-10-27.

## Context
V1 stamped camera frames with `time.monotonic_ns()` after `wait_for_frames()` returned (exposure + transport + queue latency, typically tens of ms), polled the RM65 TCP pose at 50 Hz, paired by nearest neighbour with a 50 ms limit, and **wrote no timestamps to disk**. With a moving camera at 0.1 m/s, 30 ms of error is 3 mm, larger than the accuracy goal.

## Decision
1. **Canonical clock:** `HOST_MONOTONIC` (Linux `CLOCK_MONOTONIC`, ns). `timestamp_ns` in every message is the estimated **mid-exposure** time in this domain.
2. **Record raw values from every clock, decide later.** Per camera frame the recorder stores: host arrival time, RealSense `get_timestamp()` and its domain, metadata `SENSOR_TIMESTAMP`/`FRAME_TIMESTAMP` (µs, device clock) and `ACTUAL_EXPOSURE` when available, and per-stream frame numbers. A host `CLOCK_REALTIME`/`CLOCK_MONOTONIC` pair is sampled at session start and end to convert `RS_GLOBAL` (system-time ms) to host monotonic.
3. **Mapping device to host time** (P1 S4): fit `t_host = a + b * t_device` with a lower-envelope (minimum-latency) robust fit over the session; mid-exposure = sensor timestamp mapped to host (or frame timestamp minus half exposure if the sensor timestamp is missing). Residuals are reported, and the fallback "arrival minus constant latency" must be declared in `PoseSyncInfo.method`.
4. **Record every robot sample**, not only paired ones: `robot.jsonl` stores sequence number, host time, joint angles and controller TCP pose for all samples. Pairing is a replay-time function, so the sync algorithm can change without re-recording.
5. **Robot rate:** prefer RealMan UDP real-time state push (target ≥ 100 Hz, to be verified on the installed controller/SDK); fall back to polling (stamp = midpoint of request). The source used is recorded.
6. **Interpolation, not nearest:** `T_base_tcp(t)` = slerp(R) + lerp(t) between the bracketing samples, valid only when the bracket gap ≤ `robot.max_interp_gap_ns` (placeholder default 25 ms, to be set from measured sample jitter). No extrapolation: outside the bracket the frame gets `SyncStatus.GAP_TOO_LARGE` / `OUT_OF_RANGE` and no camera prior.
7. **Residual camera–robot offset** `dt_robot_cam` is estimated from R-CALIB sessions (minimize board reprojection error over `dt`) and stored in `CalibrationBundle`; it is applied at replay time.

## Consequences
V1 sessions lack timestamps and are not reused (ADR-0007); the contracts therefore have no untimed or nearest-neighbour pairing mode.
