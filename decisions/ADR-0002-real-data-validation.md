# ADR-0002: Real recorded data is the only validation dataset
Status: Accepted. Date: 2026-10-27.

## Context
All hardware is available (wrist RealSense, RealMan RM65, GPU host). The owner does not want algorithm quality judged on synthetic renderings. The original plan used synthetic trajectories as the main fixture and ground-truth (GT) source.

## Decision
1. **No synthetic datasets.** Accuracy, robustness and gate evidence come only from sessions recorded with the real rig (`contracts/SESSION_FORMAT.md`) and replayed deterministically.
2. **Analytic unit tests stay.** Pure math (SE(3) identities, Jacobian finite differences, serialization round-trips, unit conversions) is tested with hand-written or seeded random values. These are tests of code, not datasets, and never count as accuracy evidence.
3. **SDK test doubles only for control flow.** Stubbed RealSense/RealMan objects may test error paths (disconnect, bad depth scale, read failure); never as accuracy or timing evidence.
4. **Fixture registry.** `fixtures/registry.yaml` lists every session used by tests or gates: `session_id`, scenario tags, GT type, checksum of the manifest. Raw data lives outside git under `$MVSR_DATA_ROOT`. A test that needs a missing fixture reports `SKIPPED: fixture <id> missing` explicitly; a gate with skipped fixtures is BLOCKED, never PASS.
5. **Ground truth from the real rig** (scenario codes are used by `INTEGRATION_PLAN.md`):
   - **R-STATIC** object fixed on the table, camera moved by the robot. `T_base_obj` is unknown but constant; deviation of estimated `T_base_obj(t)` from its session median measures tracking + hand-eye + timing error jointly. All R-STATIC sessions are newly recorded with the V2 recorder; V1 data is not reused (ADR-0007).
   - **R-MARKER** object rigidly attached to a fiducial plate. Board PnP gives independent `T_cam_marker(t)`; `T_marker_obj` is a constant solved once per session (fixed-gauge alignment). Board pixels are excluded from the object mask so the GT never feeds the tracker. The available plain checkerboard (11×8 inner corners; odd×even, so no 180° ambiguity) is detected only when all 88 corners are visible. Two accepted setups: (a) object and checkerboard mounted **side by side** on one rigid plate so the object never covers the board, with motion limited to keep the whole board in view; or (b) a printed ChArUco board, which tolerates partial occlusion and larger rotations. One of the two must be built before P3 T7.
   - **R-HANDHELD / R-BOTH** object moved by hand with robot static / moving. No pose GT; evaluated by held-out surface residual, R-MARKER-style board when attached, and loss/recovery statistics.
   - **R-CALIB** static 11×8 checkerboard, one take at the reachable viewpoints and speed: hand-eye check inside that envelope (P1). Camera–robot time offset is not estimated (ADR-0004).
   - **Shape GT** an object of independently known geometry (machined part, gauge block, calibrated sphere, or third-party scan). Required before any mm-level shape claim. Interim (owner currently has only a checkerboard): the board itself is a geometric reference — planarity residual and measured square pitch (caliper-verified) give depth bias / scale checks, not full 3D shape accuracy.

## Consequences
- P0 gate evidence uses sessions the owner records with the V2 recorder (ADR-0007).
- Some metrics (absolute `T_cam_obj` for a free-moving object without a board) have no GT; reports must say so instead of substituting self-consistency.
- Data acquisition sessions become planned deliverables with their own checklist (P1, P10).
