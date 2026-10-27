# Session Format (schema_version 1)

A session is one continuous recording from the real rig by `mvsr record` (ADR-0007). Raw content is immutable once the manifest status is `COMPLETE`. Derived products (masks, features, maps) live under `derived/` or in a separate run directory and never modify raw files. A session holds one or more camera streams: `wrist` (always) and `head` (optional).

## Layout
```
$MVSR_DATA_ROOT/sessions/<session_id>/      # MVSR_DATA_ROOT defaults to ./data (ADR-0006)
├── manifest.json                 # session metadata, file checksums, status
├── calibration/
│   ├── handeye_tf.txt            # verbatim copy of the reference hand-eye file (ADR-0007)
│   ├── wrist.json                # CalibrationBundle for camera_id=wrist
│   ├── head.json                 # CalibrationBundle for camera_id=head (if enabled)
│   └── <camera_id>_realsense.json# device info, stream profiles, SDK intrinsics/extrinsics, options
├── cameras/<camera_id>/
│   ├── frames.jsonl              # one record per frame_id of this stream (including dropped)
│   ├── color/<frame_id:06d>.png  # uint8 RGB, lossless (ADR-0003)
│   └── depth/<frame_id:06d>.png  # uint16 raw device units (ADR-0003)
├── robot.jsonl                   # every robot sample (not only paired ones)
├── events.jsonl                  # operator actions, scenario annotations, warnings, errors
└── derived/
    └── masks/<camera_id>/<mask_revision>/<frame_id:06d>.png  # 0/255, plus masks.jsonl with scores
```

## manifest.json
Required keys: `schema_version`, `session_id`, `status` (`RECORDING | COMPLETE | ABORTED`), `created_utc`, `host`, `software` (package version + git hash + dirty flag), `robot` (model, endpoint, SDK version, state source `udp_push|poll`, nominal rate), `cameras` (per `camera_id`: model, serial, firmware, mount `wrist|fixed`, `calibration_id`, `depth_units_m`, `color_codec`, resolution, fps, formats, device options/preset, emitter/laser, exposure mode, post-filters), `clock` (host realtime/monotonic pairs at start and end), `scenario` (tags from ADR-0002, object name, GT type, operator notes), `counts` (per camera frames seen / written / dropped; robot samples / failures), `disk` (free bytes at start/end), `files` (relative path → sha256 for every non-image file; per-image checksums live in `frames.jsonl`).

## cameras/<camera_id>/frames.jsonl
`frame_id`, `status` (`WRITTEN | DROPPED_QUEUE_FULL | DROPPED_INCOMPLETE | DROPPED_DISK | DROPPED_OPERATOR`), `t_host_arrival_ns`, `rs_timestamp_ms`, `rs_timestamp_domain`, `rs_sensor_timestamp_us`, `rs_frame_timestamp_us`, `exposure_us`, `rs_frame_number` (per stream), `color` / `depth` (relative path + sha256) when written.
No pose is stored per frame: pose pairing is computed at replay from `robot.jsonl` (ADR-0004).

## robot.jsonl
`seq`, `t_host_ns`, `source`, `joints_rad[6]`, `tcp_xyzrpy[6]` (controller units: m, rad), `read_latency_ns`, `error` (null or message). Failed reads are logged as records with `error` set.

## events.jsonl
`t_host_ns`, `type` (`START | PAUSE | RESUME | STOP | ROI_SELECTED | ANNOTATION | WARNING | ERROR`), `payload`.

## V1 data
Not imported (ADR-0007). V1 sessions remain readable in `ref/MVStreamRefine` for manual comparison only.

## Fixture registry
`fixtures/registry.yaml` (in git) lists sessions used by tests and gates:
```yaml
- session_id: 20261028T101500_CW20175_a3f1
  manifest_sha256: <hex>
  cameras: [wrist]
  scenario: [R-STATIC]
  gt: constant_T_base_obj
  object: <name>
  used_by: [P0-gate, I0, I2]
```
Tests locate data via `$MVSR_DATA_ROOT`; a missing or checksum-mismatched fixture is an explicit SKIP (unit level) or BLOCKED (gate level).
