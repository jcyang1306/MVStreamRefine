# P2 — Observation

**Dependencies:** P1
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Does not estimate final SE3 pose; output ObjectObservation and Correspondence. Avoid using mask holes as invented depth.

Code: `src/mvsr/observation/`, settings `configs/observation.yaml`, SAM 2.1 inference package vendored at `third_party/sam2-inference` (ADR-0006 amendment).

## Task sequence

- [x] **O1 — Segmentation adapter.** `observation/segmentation.py` drives `sam2_inference.StreamTracker` (public API only, lazy torch import) over the frames `ReplaySource` emits; prompt = xyxy box and/or points on one frame (`mvsr segment --box …` or `--select` for an interactive ROI in the `rig` service). Offline cache `derived/masks/<camera>/<mask_revision>/` (`observation/masks.py`): 0/255 PNG + `masks.jsonl` (status, area, score, bbox, sha256) + `revision.json` written last; `mask_revision = sam2.1-<model>@<12 hex of the parameter hash>` (model, checkpoint sha256, prompt, session manifest sha256, frame list). Status `OK | LOW_CONFIDENCE | EMPTY`; confidence = mean sigmoid(logit) inside the mask (the video predictor has no IoU score). Frames before the prompt frame are not segmented.

- [x] **O2 — Depth preprocessing.** `observation/depth.py`: mask eroded `mask_erode_px` ∩ depth in [0.07, 0.60] m ∩ no 4-neighbour discontinuity (`edge_jump_rel` of z, removes hole borders and flying pixels) ∩ 3×3 std ≤ 3 σ(z), σ(z) = a + b z². Holes are never filled. Points `z · [x, y, 1]` with undistorted rays (`calib/intrinsics.py`, shared with the P1 board check); normals from central differences, facing the camera. Registration check: signed offset of the occluding depth step from the colour mask edge (`silhouette_offsets_px`). I1 used 3 px / 0.02 / a = 1 mm, b = 0.012 m⁻¹; retuned 2026-10-30 to 7 px / 0.014 / a = 0, b = 0.0224 m⁻¹ (`configs/provisional_params.yaml`).

- [x] **O3 — Feature extraction.** `observation/features.py`: `FeatureExtractor` protocol (the adapter boundary for SuperPoint/DISK etc.; none shipped, other `features.type` values are a config error) and the ORB baseline inside the mask eroded 6 px; raw (distorted) pixel coordinates and 32-byte descriptors stored as detected.

- [x] **O4 — Long-lived tracks.** `observation/tracks.py`: KLT with forward-backward check, statuses `NEW | TRACKED | OCCLUDED | RETIRED`, confidence `1 − fb/fb_max`, occlusion budget 5 frames, ids never reused, re-detection to 400 visible tracks, retirement when the descriptor drifts > 64 bits from its birth descriptor (tuned, see evidence).

- [x] **O5 — Correspondences.** `observation/correspondence.py`: candidates by shared track id or ORB (ratio 0.8, mutual best, Hamming ≤ 64); screening by a seeded rigid RANSAC on 3D-3D pairs (`ransac_rigid_m`, now 9 mm), one-sided pairs by reprojection (`ransac_reproj_px`, 7.5 px), depth-less pairs by Sampson distance, essential-matrix fallback; rejected matches kept with `RIGID_RESIDUAL | REPROJECTION | EPIPOLAR | NO_MODEL`. The screening model is internal; no pose is output.

- [x] **O6 — Quality benchmark.** `observation/benchmark.py` (`mvsr observe-report`, `mvsr i1-gate`): R-STATIC reference `T_cj_ci = inv(T_base_cam_j) T_base_cam_i`; mask agreement, match precision/recall and inlier 3D residuals per frame gap, track life and reprojection drift by age, depth coverage, silhouette offset, determinism. Textureless object: the cup (pink, almost no texture). Reflective objects are dropped from the plan and R-HARD scenes are out of scope (owner 2026-10-30).

## P2.1 — Planned follow-up (owner 2026-10-30; not implemented)

Baseline for every item: mask revision `sam2.1-tiny@7558ff4ba813` (fp32, full frame, every frame; SAM 323 ms/frame on the RTX 3060) on the cup take `…9fb3`. Each item adds a config switch whose default keeps today's behaviour, writes a new mask revision (the switch enters the revision hash), and is accepted only if `mvsr i1-gate` still passes.

**Implementation order (do not skip):** O7 → O8 → O10 → O9. O7 is independent. O8 freezes the input geometry that `torch.compile` (O10) and the P9 TensorRT encoder (RT7) require. O9 changes *when* SAM runs, not how. TensorRT is P9 RT7, after O8.

- [ ] **O7 — bf16 inference.** `sam2.precision: fp32 | bf16`; wrap `add_prompt`/`track` in `torch.autocast("cuda", dtype=torch.bfloat16)` (matmul/conv in bf16, softmax/norm kept fp32 by autocast). Measure ms/frame and peak VRAM. Accept: per-frame IoU vs. the fp32 revision median ≥ 0.99, p5 ≥ 0.97.
- [ ] **O8 — ROI segmentation.** `sam2.roi: {enabled, margin, image_size}`: crop the previous mask bbox enlarged by `margin` (default 0.4), feed the crop at a reduced model input (`image_size` 512, `build_sam2` override), paste the result back. SAM2 memory assumes a fixed frame geometry, so the ROI is held fixed for a window and re-prompted from the previous mask when it must move; if the mask touches the ROI border, enlarge or fall back to the full frame for that frame. Record ROI and fallback per frame in `masks.jsonl`. Accept: IoU vs. full-frame revision median ≥ 0.98, p5 ≥ 0.95; ms/frame reported against O7.
- [ ] **O9 — Lower SAM rate + mask propagation.** `sam2.refresh: {every_n, propagate: flow | depth_pose, triggers}`: SAM on every N-th frame; in between, warp the last mask by (a) dense optical flow or (b) back-projecting the masked depth and re-projecting with the relative camera/object motion (robot prior for a static object, P3 pose otherwise), followed by morphological closing. Run SAM early when the propagated mask changes area by > x %, when > y % of its pixels lie at background depth, or when the flow forward-backward error is high. `masks.jsonl` gets `source: sam | propagated` and the trigger. Accept at N = 5: IoU of propagated frames vs. SAM-every-frame median ≥ 0.95, p5 ≥ 0.90, SAM calls ≤ 25 %.
- [ ] **O10 — Compiled inference.** Expose `vos_optimized` (SAM2's `torch.compile` of image encoder, memory attention and mask decoder) through the vendored `sam2_inference` builder (`sam2.compile: false | vos_optimized`); record compile/warm-up time, steady ms/frame, and whether repeated runs give identical masks. Requires fixed input size (after O8). Accept: same IoU bounds as O7. TensorRT export belongs to P9 RT7.
- [ ] **O11 — Track quality.** (a) birth filter on the KLT minimum eigenvalue (`tracks.min_eig`), threshold from data: smallest value at which median drift at age ≤ 30 frames stops improving while median tracked keypoints ≥ 2 × the I1 minimum; (b) geometric-consistency retirement: a track whose 3D residual to the per-pair rigid model exceeds `k·σ(z)` for `m` consecutive frames is retired; (c) tracks feed short baselines only, long baselines use keyframe descriptor matches (ADR-0008). Accept: drift at age > 100 frames on the full take moves toward the ~7 px reference floor without dropping below the I1 track minimum.
- [ ] **O12 — Evaluation resolution.** Precision as a curve over 2 / 4 / 6 / 10 mm, planted-outlier rejection rate, RANSAC threshold scaled by `σ(z)` of both points; GT finer than the robot reference needs an R-MARKER take.

`ObservationBuilder(masks, bundle, cfg).process(SensorFrame) -> ObjectObservation` (contract port). `quality` carries `mask_area_px, valid_depth_ratio, n_keypoints, mask_score, mask_status, geometry_ratio, keypoints_with_depth, tracks_new/tracked/occluded/retired, depth{…}`; `mask_prob` is None.

## Gate evidence (I1, 2026-10-30)
Operating scope (owner 2026-10-30): object always in view, slow motion; occlusion, fast motion and missing depth are not validated. Session `20261028T080558_CW20175_9fb3` (R-STATIC cup, registered `used_by: I1`), mask revision `sam2.1-tiny@7558ff4ba813` (prompt box `580 25 762 275` on frame 0, confirmed by the owner), evaluated with the newest hand-eye (`configs/sensors.yaml: replay.handeye`), run `data/runs/20261030T081416_CW20175_ad85`. Reproduce: `docker compose run --rm dev mvsr i1-gate`. Mask overlays: `data/runs/p2_mask_overlay.png`.

**Decision: PASS** (all hard checks and all placeholder targets):

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

**Parameters retuned 2026-10-30** from the same two takes (`mvsr derive-params`, runs `…090635_d2f1` sensors, `…091800_7786` observation): σ(z) = 0.0224 z² (a = 0, valid 0.33–0.41 m), `mask_erode_px` 7, `edge_jump_rel` 0.014, correspondence 9 mm / 7.5 px / σ_px 1.15. KLT `fb_max_px` 1.0 and `max_descriptor_drift_bits` 64 stay at the I1 values (a 400-frame sweep preferred 0.5 / 48 on age ≤ 30 drift only). All of these are in `configs/provisional_params.yaml`; `mvsr params-status` and `mvsr register-session` remind when a new matching session is registered.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run the matching I-gate in `../INTEGRATION_PLAN.md` by replaying registered real sessions (`fixtures/registry.yaml`); live-hardware checks are additional, marked `hardware`.
- Performance/accuracy log: include relevant latency, failure cases, quality and version numbers.

## Completion checklist
- [x] API documentation and schema tests complete (module docstrings; `test_masks`, `test_segmentation`, `test_observation_depth`, `test_features_tracks`, `test_correspondence`, `test_observation_builder`).
- [x] Deterministic offline test passes and reproduction command recorded (`docker compose run --rm dev tools/check.sh`: 198 passed, 3 hardware skipped; `tests/test_i1_sessions.py`, `tests/test_sam2_real.py` on real data with the GPU).
- [x] Integration gate evidence and quantitative metrics saved (I1 PASS, 2026-10-30).
- [x] Known limitations recorded; no false success or hidden fallback.
- [x] Handoff report names changed files, tests, residual risks and next phase (below).

## Handoff (P2 → P3)
**Shipped:** `src/mvsr/observation/{config,masks,segmentation,depth,features,tracks,builder,correspondence,benchmark,report}.py`, `src/mvsr/calib/intrinsics.py` (rays, moved from the P1 board module), `src/mvsr/io/images.py` mask PNG IO, `src/mvsr/yaml_config.py` list support, `configs/observation.yaml`, `third_party/sam2-inference`, Docker `torch-deps` stage, CLI `mvsr segment | masks | observe-report | i1-gate`, tests above.

**Owner decisions (2026-10-30):** operating scope is object-in-view and slow motion (no R-HARD); no reflective objects; prompt box of the cup take confirmed; evaluations use the 2026-10-28 sessions and the newest hand-eye. In the realtime system the user draws the box while robot and object are still (subplans/09_REALTIME.md RT4).

**Residual risks:** SAM 2.1 at 2.9 fps is offline-only (P9 needs bf16/smaller input/TensorRT); KLT tracks on the low-texture cup still drift on long baselines (14 px median after 100 frames) — P3 should prefer short baselines or re-verify with descriptors; precision is measured against a reference with ~4 mm error, so sub-reference outliers are invisible; occlusion / fast motion / missing depth are outside the validated scope; the optional SAM CUDA hole-filling extension is not compiled (post-processing only); hardware tests were not re-run after the image rebuild (rig was offline).

**Next:** P3 `subplans/03_TRACKING.md` consumes `ObjectObservation` + `CorrespondenceSet` and must handle `EMPTY` masks, frames without priors, and NaN keypoint depth.
