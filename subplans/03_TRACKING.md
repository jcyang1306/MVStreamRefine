# P3 — Fast 6D Tracking

**Dependencies:** P2
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Outputs PoseEstimate; avoid implying camera robot pose reveals arbitrary free-moving object pose.

Keyframe-anchored tracking (ADR-0008): the live frame is registered against the local keyframe set / object map, not chained to the previous frame. That bounds pose error by distance to the nearest keyframe. It is **not** bundle adjustment — P6 jointly refines keyframe poses and landmarks later. Do not add BA, loop closures, or a separate "drift correction" step in this phase.

## Task sequence

- [x] **T1 — Reference-frame initialization.** `tracking/tracker.py`: the first accepted observation with a robot prior and sufficient geometry fixes the ADR-0005 gauge; `T_base_obj_init` is persisted in every tracking report and never recentered.

- [x] **T2 — Robot-assisted prediction.** Constant-velocity base-object prediction from the last two valid estimates (bounded against discontinuities), converted by the current camera prior. The previous frame is used **only** for prediction / initialization (ADR-0008).

- [x] **T3 — Pose hypotheses (keyframe-anchored).** `tracking/session.py` supplies a deterministic bounded in-memory anchor window until P4; the current observation is matched to local anchors by ORB and to the latest anchor by short-baseline tracks, then P2 rigid RANSAC gives `T_current_anchor`. The previous frame is never a measurement anchor. Every result records `anchor_keyframe_id`, method and hypothesis count.

- [x] **T4 — Geometry refinement.** `tracking/icp.py`: deterministic coarse-to-fine point-to-plane ICP against anchor/map geometry, distance + normal gates, Huber weights and covariance. On the low-texture cup, unconstrained ICP worsened 100-frame static p95 from ~14 mm to ~48 mm, so a refinement beyond 1 mm / 0.5° is rejected in favour of the feature hypothesis. **Known limit:** the full take still accumulates keyframe-chain drift (T7); P4/P6 evidence is required before mapping.

- [x] **T5 — Validity & degeneracy.** `PoseEstimate.quality` includes inliers, overlap, RMSE, normalized Hessian minimum eigenvalue, hypothesis count, correction from prediction and ICP correction. Low-rank ICP, competing separated hypotheses, low support and large correction produce explicit reason codes; identity is never a failure pose.

- [x] **T6 — State machine.** INITIALIZING, TRACKING, DEGRADED, LOST, RELOCALIZING with bounded prediction-only tolerance. Invalid observations carry `T_cam_obj=None`; only VALID poses may become temporary anchors. P3 has no map-commit API.

- [x] **T7 — Validation implementation.** `mvsr tracking-report | i2-gate` reports R-STATIC spread, status/reasons, drift vs. anchor age/distance, deterministic pose digest and frame-chain / ICP-only / robot-prior-only ablations. R-MARKER and R-BOTH metrics activate when those sessions are registered. R-HARD is out of scope.

## Gate evidence (I2 partial, 2026-11-05)
Session `20261028T080558_CW20175_9fb3`, full 1090 emitted frames, mask revision `sam2.1-tiny@7558ff4ba813`; run `data/runs/20261105T044934_CW20175_98b0`. Reproduce: `docker compose run --rm dev mvsr i2-gate`.

**Decision: PASS-WITH-LIMITS.** Hard checks pass (R-STATIC present, valid ratio 0.913 ≥ 0.90, 60-frame replay digest identical), but the quality targets and required dynamic evidence do not:

| Metric | Keyframe tracker | Frame-to-frame | ICP-only |
|---|---:|---:|---:|
| `T_base_obj` translation median / p95 | 88.2 / 217.2 mm | 104.8 / 178.9 mm | 65.2 / 199.8 mm |
| `T_base_obj` rotation median / p95 | 14.9 / 34.1° | 23.6 / 33.8° | 10.1 / 29.5° |

Other evidence: 995 VALID / 94 DEGRADED / 1 INVALID; 70 anchors inserted (12 retained); anchor distance median / p95 11.2 / 60.1 mm and 2.05 / 11.40°; 2.62 fps including the primary tracker plus ablations. The keyframe tracker improves median error over frame chaining but **does not meet** the provisional 15 mm / 3° p95 target over the complete take. This is not hidden by the gate: P4 must improve viewpoint/covisibility retrieval and P6 must optimize keyframe poses; R-MARKER/R-BOTH are still missing.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run the matching I-gate in `../INTEGRATION_PLAN.md` by replaying registered real sessions (`fixtures/registry.yaml`); live-hardware checks are additional, marked `hardware`.
- Performance/accuracy log: include relevant latency, failure cases, quality and version numbers.

## Completion checklist
- [x] API documentation and schema tests complete.
- [x] Deterministic offline test passes and reproduction command recorded.
- [x] Integration gate evidence and quantitative metrics saved (I2 PASS-WITH-LIMITS).
- [x] Known limitations recorded; no false success or hidden fallback.
- [x] Handoff report names changed files, tests, residual risks and next phase (below).

## Handoff (P3 → P4)
**Shipped:** `src/mvsr/tracking/{config,icp,tracker,session,benchmark,report}.py`, `configs/tracking.yaml`, CLI `mvsr tracking-report | i2-gate`, unit tests and `tests/test_i2_sessions.py`.

**Residual risks / blockers:** full-take drift misses the target by a large margin; the temporary P3 anchor window keeps the gauge plus recent anchors but is not P4's viewpoint/covisibility memory; the cup is both low-texture and geometrically weak for ICP; hand-eye/robot reference is not independent pose GT; no R-MARKER or R-BOTH session exists. Do not start committed TSDF mapping from these poses.

**Next:** P4 implements persistent, viewpoint-diverse keyframe memory and retrieval, then rerun I2. Record R-MARKER and R-BOTH before claiming complete I2; P6 BA remains responsible for residual keyframe-graph inconsistency.
