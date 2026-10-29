# P2 — Observation

**Dependencies:** P1
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Does not estimate final SE3 pose; output ObjectObservation and Correspondence. Avoid using mask holes as invented depth.

Code: `src/mvsr/observation/`, settings `configs/observation.yaml`, SAM 2.1 inference package vendored at `third_party/sam2-inference` (ADR-0006 amendment).

## Task sequence

- [x] **O1 — Segmentation adapter.** `observation/segmentation.py` drives `sam2_inference.StreamTracker` (public API only, lazy torch import) over the frames `ReplaySource` emits; prompt = xyxy box and/or points on one frame (`mvsr segment --box …` or `--select` for an interactive ROI in the `rig` service). Offline cache `derived/masks/<camera>/<mask_revision>/` (`observation/masks.py`): 0/255 PNG + `masks.jsonl` (status, area, score, bbox, sha256) + `revision.json` written last; `mask_revision = sam2.1-<model>@<12 hex of the parameter hash>` (model, checkpoint sha256, prompt, session manifest sha256, frame list). Status `OK | LOW_CONFIDENCE | EMPTY`; confidence = mean sigmoid(logit) inside the mask (the video predictor has no IoU score). Frames before the prompt frame are not segmented.

- [x] **O2 — Depth preprocessing.** `observation/depth.py`: mask eroded 3 px ∩ depth in [0.07, 0.60] m ∩ no 4-neighbour discontinuity (> 2 % of z, removes hole borders and flying pixels) ∩ 3×3 std ≤ 3 σ(z), σ(z) = 1 mm + 0.012 z². Holes are never filled. Points `z · [x, y, 1]` with undistorted rays (`calib/intrinsics.py`, shared with the P1 board check); normals from central differences, facing the camera. Registration check: signed offset of the occluding depth step from the colour mask edge (`silhouette_offsets_px`).

- [x] **O3 — Feature extraction.** `observation/features.py`: `FeatureExtractor` protocol (the adapter boundary for SuperPoint/DISK etc.; none shipped, other `features.type` values are a config error) and the ORB baseline inside the mask eroded 6 px; raw (distorted) pixel coordinates and 32-byte descriptors stored as detected.

- [x] **O4 — Long-lived tracks.** `observation/tracks.py`: KLT with forward-backward check, statuses `NEW | TRACKED | OCCLUDED | RETIRED`, confidence `1 − fb/fb_max`, occlusion budget 5 frames, ids never reused, re-detection to 400 visible tracks, retirement when the descriptor drifts > 64 bits from its birth descriptor (tuned, see evidence).

- [x] **O5 — Correspondences.** `observation/correspondence.py`: candidates by shared track id or ORB (ratio 0.8, mutual best, Hamming ≤ 64); screening by a seeded rigid RANSAC on 3D-3D pairs (6 mm), one-sided pairs by reprojection, depth-less pairs by Sampson distance, essential-matrix fallback; rejected matches kept with `RIGID_RESIDUAL | REPROJECTION | EPIPOLAR | NO_MODEL`. The screening model is internal; no pose is output.

- [x] **O6 — Quality benchmark.** `observation/benchmark.py` (`mvsr observe-report`, `mvsr i1-gate`): R-STATIC reference `T_cj_ci = inv(T_base_cam_j) T_base_cam_i`; mask agreement, match precision/recall and inlier 3D residuals per frame gap, track life and reprojection drift by age, depth coverage, silhouette offset, determinism. Textureless object: the cup (pink, almost no texture). **Reflective object and R-HARD sessions are not recorded yet.**

`ObservationBuilder(masks, bundle, cfg).process(SensorFrame) -> ObjectObservation` (contract port). `quality` carries `mask_area_px, valid_depth_ratio, n_keypoints, mask_score, mask_status, geometry_ratio, keypoints_with_depth, tracks_new/tracked/occluded/retired, depth{…}`; `mask_prob` is None.

## Gate evidence (I1, 2026-10-29)
Session `20261028T080558_CW20175_9fb3` (R-STATIC cup, registered `used_by: I1`), mask revision `sam2.1-tiny@7558ff4ba813` (prompt box `580 25 762 275` on frame 0, chosen from the first image — owner to confirm), run `data/runs/20261029T091903_CW20175_7fe6`. Reproduce: `docker compose run --rm dev mvsr i1-gate`. Mask overlays: `data/runs/p2_mask_overlay.png`.

**Decision: BLOCKED** — the only failing check is the hard requirement of an R-HARD session (occlusion, invalid depth, object leaving frame). Everything measurable on R-STATIC passes:

| Metric | Value |
|---|---|
| frames with a cached mask | 1090 / 1090, all `OK`, score median 0.995 |
| SAM 2.1 tiny speed (RTX 3060, fp32) | 2.9 fps offline (323 ms model, 32 ms read+verify per frame) |
| mask agreement under the robot reference (gap 1 … 30) | median 1.00 |
| valid depth in mask / geometry ratio | median 0.88 / 0.77 |
| silhouette depth-step offset from colour edge | median +1.5 px, p95 5.5 px (stereo edge fattening; removed by the 3 px erosion) |
| visible tracks per frame | median 104 (88 with depth) |
| track life | median 34 frames, 29 % ≥ 100 frames |
| track reprojection drift vs. reference, age ≤ 1 / ≤ 30 / > 100 frames | median 1.4 / 6.2 / 14.0 px |
| inliers per pair, gap 5: track / ORB | 58.5 / 39 |
| inlier precision (1 cm or 8 px), gap 5 … 30 | 1.00 (threshold above the reference noise, so not discriminative) |
| recall, gap 5: track / ORB | 0.78 / 0.84 |
| inlier 3D residual vs. reference, gap 5: track / ORB | median 2.5 / 2.1 mm, p95 6.7 / 6.1 mm |
| determinism (60 frames, two independent builders) | identical |

Tuning of `tracks.max_descriptor_drift_bits` on the first 500 frames (runs `…090837_ec21` off, `…090933_0df3` 64, `…091029_9462` 80): drift of tracks older than 100 frames 13.2 → 6.7 px median, visible tracks 170 → 120. On the full session (baseline run `…090531_487a` without the limit) the same limit takes old-track drift from 27.1 to 14.0 px median, still above the ~7 px reference floor.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run the matching I-gate in `../INTEGRATION_PLAN.md` by replaying registered real sessions (`fixtures/registry.yaml`); live-hardware checks are additional, marked `hardware`.
- Performance/accuracy log: include relevant latency, failure cases, quality and version numbers.

## Completion checklist
- [x] API documentation and schema tests complete (module docstrings; `test_masks`, `test_segmentation`, `test_observation_depth`, `test_features_tracks`, `test_correspondence`, `test_observation_builder`).
- [x] Deterministic offline test passes and reproduction command recorded (`docker compose run --rm dev tools/check.sh`: 198 passed, 3 hardware skipped; `tests/test_i1_sessions.py`, `tests/test_sam2_real.py` on real data with the GPU).
- [ ] Integration gate evidence and quantitative metrics saved — R-STATIC part saved; **BLOCKED on an R-HARD recording**.
- [x] Known limitations recorded; no false success or hidden fallback.
- [x] Handoff report names changed files, tests, residual risks and next phase (below).

## Handoff (P2 → P3)
**Shipped:** `src/mvsr/observation/{config,masks,segmentation,depth,features,tracks,builder,correspondence,benchmark,report}.py`, `src/mvsr/calib/intrinsics.py` (rays, moved from the P1 board module), `src/mvsr/io/images.py` mask PNG IO, `src/mvsr/yaml_config.py` list support, `configs/observation.yaml`, `third_party/sam2-inference`, Docker `torch-deps` stage, CLI `mvsr segment | masks | observe-report | i1-gate`, tests above.

**To unblock I1:** record one R-HARD take (hand occluding the object, object leaving and re-entering the view, a fast motion, some invalid-depth surface), `mvsr segment <id> --select`, register it `used_by: I1`, run `mvsr i1-gate`.

**Residual risks:** SAM 2.1 at 2.9 fps is offline-only (P9 needs bf16/smaller input/TensorRT); KLT tracks on the low-texture cup still drift on long baselines (14 px median after 100 frames) — P3 should prefer short baselines or re-verify with descriptors; precision is measured against a reference with ~4 mm error, so sub-reference outliers are invisible; no reflective-object or R-HARD evidence; the optional SAM CUDA hole-filling extension is not compiled (post-processing only); hardware tests were not re-run after the image rebuild (rig was offline).

**Next:** P3 `subplans/03_TRACKING.md` consumes `ObjectObservation` + `CorrespondenceSet` and must handle `EMPTY` masks, frames without priors, and NaN keypoint depth.
