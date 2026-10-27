# ADR-0005: Object frame (gauge) definition
Status: Accepted. Date: 2026-10-27.

## Decision
At the first accepted keyframe `k0` of a session (tracking T1):
- **Origin** = component-wise median of the valid `geometry_mask` 3D points of `k0`, expressed in `base`: `c_base`.
- **Orientation** = axes parallel to the robot `base` axes at that instant (gravity-aligned, z up if base z is up).
- Hence `T_base_obj_init = [I | c_base]` and `T_cam_obj(k0) = inv(T_base_cam_prior(k0)) @ T_base_obj_init`.
- `T_base_obj_init` is stored once in the session map metadata and is **never changed**, even if later optimization corrects `T_base_cam` or `T_cam_obj(k0)`. Subsequent corrections move poses, not the gauge.
- Exports may use `T_object_export` (e.g. bounding-box centered); it is a separate constant written next to the mesh.

## Rationale
- Gravity-aligned axes make meshes readable and keep TSDF volume bounds compact around the origin.
- For a static object (R-STATIC), the object frame differs from the base frame by a pure known translation, so a robot-pose-only base-frame TSDF baseline is directly comparable.
- The definition is deterministic and depends only on recorded data.

## Consequences
If `k0` has a wrong mask, the origin may be off the object; this affects readability only, not correctness. The gauge must be fixed (not optimized) in BA (P6 B4).
