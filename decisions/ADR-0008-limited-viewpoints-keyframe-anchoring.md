# ADR-0008: Limited viewpoint coverage, no loop closure, keyframe-anchored tracking
Status: Accepted (owner 2026-10-30). Date: 2026-10-30.

## Context
The arm's mounting pose limits the reachable wrist viewpoints: the camera sees one side / an arc of the object and cannot orbit it. A camera path that comes back to a previously seen side after a long excursion (the usual SLAM loop closure) does not happen. Operating scope (owner 2026-10-30): the object stays in view and moves slowly. P2 measured frame-to-frame KLT tracks drifting with age (median 14 px after 100 frames on the low-texture cup) while the robot reference floor is ~7 px.

## Decision
1. **No loop-closure dependency.** No phase may rely on loop closures to remove drift. P6 B7 keeps a keyframe graph without loop edges; any revisit inside the reachable arc is handled as ordinary keyframe re-matching.
2. **Keyframe-anchored tracking.** The current frame is registered against a local set of keyframes and the local object map (P3 T3/T4, P4 K3), not chained to the previous frame. The previous frame (and the robot prior) only predicts / initializes. Accumulated error then grows with the distance to the nearest keyframe, not with the number of frames.
3. **Short-baseline tracks, keyframe descriptors for long baselines.** P2 KLT tracks are trusted only over short baselines; associations across keyframes come from descriptor matching against keyframes (P2 O5 `orb` method) with geometric screening. P6 landmarks are built from keyframe matches, not from long KLT chains.
4. **Absolute anchors.** The object gauge is fixed at the first keyframe (ADR-0005). When the object is static in `base`, the robot prior is an absolute camera anchor (like GPS/IMU in SLAM) and may weight BA. It never constrains independent object motion (P6 purpose).
5. **Partial models are the expected output.** Mapping completeness is reported relative to the reachable surface, not the full object.

## Drift: what P2/P3 must do vs what waits for P6
P6 bundle adjustment jointly refines keyframe poses and landmarks. That does **not** mean P2/P3 can ignore drift:

- **P2/P3 must not chain the object pose to the previous frame.** Frame-to-frame KLT already drifts (median 14 px after 100 frames on the cup). If tracking commits associations along that chain, P6 cannot recover a consistent object frame — BA can only refine the graph it is given.
- **P2/P3 must not implement BA, pose-graph loop closures, or a "drift correction" feature.** Those belong in P6. Tracking's job is to keep each live pose close to an existing keyframe (and to insert a new keyframe when overlap with the anchor drops).
- **P6 is not a substitute for keyframe anchoring, and it will not close a loop the arm cannot fly.** It reduces residual inconsistency among covisible keyframes inside the reachable arc. Absolute camera anchors (robot prior while the object is static in `base`) are the only global constraint.

## Consequences
- P3/P4 implement keyframe selection by object-frame viewpoint and covisibility, and track against them from the start.
- P6 B7 global graph has covisibility edges only; marginalization and outlier handling do not assume loops.
- P8 relocalization retrieves only within the seen arc; out-of-view re-entry is outside the current scope.
- P10 reports reconstruction coverage of the reachable surface and drift vs. distance to the anchoring keyframe.
- Completeness, texture on the unseen side, and any later full-object CAD alignment are out of scope until the cell can reach more viewpoints (a new ADR).
