# ADR-0003: Depth stored in raw device units; color stored losslessly
Status: Accepted. Date: 2026-10-27.

## Context
V1 captures read z16 depth with device scale `1e-4 m/unit` (camera serial 323622271240, consistent with a D405) and re-quantized it to uint16 millimeters for PNG. That discards the 0.1 mm resolution the sensor provides at close range, which conflicts with a 1–2 mm surface goal. V1 also stored color as JPEG q95, whose block artifacts bias feature locations and photometric residuals.

## Decision
- Depth PNG stores the **raw uint16 device value** unchanged. `depth_units_m` (meters per unit) is recorded in the session manifest and in `CalibrationBundle`; for the current camera this is `1e-4` (max 6.5535 m). Value `0` means invalid.
- In memory, `depth_m = depth_raw_u16 * depth_units_m` (float32); conversion happens once, in the session reader / sensor source. No other module scales depth.
- Each camera declares its own unit: wrist D405 `1e-4`, head D435 `1e-3` (ADR-0007).
- Color is stored as lossless PNG (fast compression level). JPEG is allowed only as an explicitly declared `color_codec` in the manifest; JPEG sessions are not valid for photometric or sub-pixel accuracy evaluation.
- Device-side RealSense post-processing filters are **off** during recording; the active preset, laser power, exposure mode and emitter state are written to the manifest. Filtering belongs to observation (P2).
- When handing depth to Open3D, the mapping wrapper passes either float meters with `depth_scale=1.0` or raw uint16 with `depth_scale=1/depth_units_m`; it must assert which one in code.

## Consequences
Roughly 3–4 MB per 1280×720 RGB-D frame; ~100 MB/s at 30 fps. The recorder uses a bounded writer queue and logs every frame it cannot persist (no silent drop). Disk budget per session must be checked before recording.
