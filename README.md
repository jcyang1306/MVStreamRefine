# MVStreamRefine V2 — staged implementation plan pack

Start at `MASTER_PLAN.md`; the API/mathematical authority is in `contracts/`; implement one `subplans/NN_*.md` at a time and pass corresponding `INTEGRATION_PLAN.md` gate. This package is a proposed rebuild plan based on the supplied **MVStreamRefine README**, not an audit or code modification of an existing repository.

## Recommended developer workflow
1. Read master + contracts + current subplan; establish branch and phase ID.
2. List planned file edits, API and fixture dependencies.
3. Implement minimal validated slice; validate on recorded real sessions (`fixtures/registry.yaml`) before live hardware runs. No synthetic datasets (ADR-0002).
4. Run unit and phase integration test and log metrics.
5. Update subplan checklist / handoff report; move to next only after gate PASS.

See `AGENT_TASK_TEMPLATE.md` for prompts to Cursor/Codex.

## P0 usage (workstation CW20175, Docker only)
```bash
mkdir -p data                                   # $MVSR_DATA_ROOT (git-ignored)
docker compose build dev rig                    # mvsr:cu121, mvsr:cu121-rig
docker compose run --rm dev tools/check.sh      # ruff, mypy --strict core, pytest

# one P0 take (robot is moved by the operator; the recorder only reads state).
# Omit --duration; stop with Q in the preview or Ctrl+C. inspect-session counts frames from the records.
xhost +si:localuser:$USER
docker compose run --rm rig mvsr record --object <name> --gt constant_T_base_obj
#   defaults: --cameras wrist,head --scenario R-STATIC, live preview on
#   preview: SPACE pause/resume, A annotate, Q/ESC stop; --no-preview if headless

docker compose run --rm dev mvsr inspect-session <session_id> [--json]   # measured duration, frame counts, rates
docker compose run --rm dev mvsr verify-session <session_id> --p0-gate   # integrity + P0 completeness (no fps target)
docker compose run --rm dev mvsr register-session <session_id> --used-by P0-gate
docker compose run --rm rig pytest -q tests/hardware --run-hardware       # 10 s live pytest smokes (auto --duration)
```
## P1 usage (sensors: clock mapping, robot priors, SensorFrame replay)
```bash
docker compose run --rm dev mvsr replay <session_id> [--speed 1.0] [--limit N]   # SensorFrames; prints output sha256
docker compose run --rm dev mvsr sensors-report <session_id> [...] [--square-mm 15]  # timing/sync (+ board for R-CALIB) -> data/runs/
docker compose run --rm dev mvsr i0-gate                                           # I0 on registry sessions used_by I0
docker compose run --rm rig pytest -q tests/hardware --run-hardware                # includes the live SensorFrame source
```
```python
from mvsr.sensors.replay import ReplaySource
for frame in ReplaySource("data/sessions/<session_id>"):   # wrist by default
    frame.header.timestamp_ns, frame.T_base_cam_prior, frame.pose_sync.status
```
Settings and placeholders: `configs/sensors.yaml`; evidence and limits: `subplans/01_SENSORS.md`.

Recorder settings, units and placeholders: `configs/record.yaml`. The hand-eye reference is
`configs/handeye/handeye_tf_20261028.txt` (from `calibration_result.json`, sha256 pinned, ADR-0007). The 2026-10-12 file remains for sessions that already copied it.

### P0 format notes (additions to `contracts/SESSION_FORMAT.md`, all optional and raw-only)
- `frames.jsonl`: `color`/`depth` carry `raw_sha256` (pixel hash before PNG encoding, used by
  `verify-session` to prove the lossless round trip); `depth_timing` (raw clocks of the depth
  frame; the contract fields come from the color frame); `rs_extra` (librealsense
  `backend_timestamp`/`time_of_arrival`, ms); `drop_detail` (e.g. `process_queue`, `paused`).
- `manifest.json`: `duration_s`, `stop_reason`, `errors`, `performance` (per-camera fps, queue
  high-water marks, MB/s, CPU), `recorder` (config snapshot + sha256), per-camera
  `depth_units_m_sdk`, `global_time_enabled`, `aligned_to`, `device_options_file`.
- `warmup_frames` framesets per camera are discarded before `START`; `frame_id` 0 is the first
  frame after warm-up. Frames during PAUSE get a `frame_id` and `DROPPED_OPERATOR`.

## Key references
- V1 reference implementation (read-only): `ref/MVStreamRefine`.
- Contracts: `contracts/DATA_CONTRACTS.md`, `contracts/COORDINATES_AND_TIMING.md`, `contracts/SESSION_FORMAT.md`.
- Decisions: `decisions/ADR-0001` (architecture) … `ADR-0007` (new recordings only, recorder in P0, head camera).
