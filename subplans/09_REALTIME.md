# P9 — Online Runtime

**Dependencies:** P3–P8
**Scope:** owned subsystem only. **Read:** `../MASTER_PLAN.md`, `../contracts/DATA_CONTRACTS.md`, `../contracts/COORDINATES_AND_TIMING.md`.

## Purpose and boundary
Offline deterministic pipeline must remain the reference; online event order and version conflict behavior tested.

## Task sequence

- [ ] **RT1 — Async graph.** Capture, segmentation, tracking, optimizer, mapper and viewer independent bounded queues.

- [ ] **RT2 — Backpressure.** Define frame drop policy, latest-state semantics, ordering, per-stage timing; no silently stale pose commits.

- [ ] **RT3 — Thread safety.** Immutable message objects, versioned optimization updates, atomic map snapshots.

- [ ] **RT4 — Lifecycle.** Start/pause/resume, loss recovery, shutdown/export, error reporting, restart from persisted keyframes. Start-up (owner 2026-10-30): robot and object stay still while the user draws the object box in the live preview; that box is the SAM prompt, then robot and object may move.

- [ ] **RT5 — Robot safety.** Optional motion commands separate module with hardware interlocks; no robot motion in tracking core.

- [ ] **RT6 — Performance.** Full latency breakdown, peak VRAM/RAM, throughput and queue lag on target GPU and CPU. Segmentation baseline: SAM 2.1 tiny fp32 323 ms/frame on the RTX 3060 (P2); offline speed-ups are P2.1 in this order: O7 bf16 → O8 ROI + smaller input → O10 `vos_optimized` (`torch.compile`) → O9 lower SAM rate with propagation. Target: segmentation keeps up with the tracker at ≥ 10 Hz effective mask rate.

- [ ] **RT7 — TensorRT image encoder.** After P2.1, export the SAM2 image encoder (fixed input size from O8) to ONNX → TensorRT FP16 engine; memory attention and mask decoder stay in PyTorch (their state is dynamic). Engine is built per GPU / TensorRT version, cached with the checkpoint sha256 and input size in its name, and rebuilt automatically on mismatch. Accept: mask IoU vs. the PyTorch path median ≥ 0.99 on the cup take, encoder latency and end-to-end ms/frame reported; fall back to PyTorch with a logged warning if the engine is missing.

## Interfaces and delivery
- Define/export the public inputs/outputs in the shared contracts; do not re-define `T_cam_obj`, depth scale, timestamp or keyframe IDs locally.
- Unit tests: normal path, corrupt/missing input, SE3 direction, deterministic replay and numerical edge cases where applicable.
- Integration: run the matching I-gate in `../INTEGRATION_PLAN.md` by replaying registered real sessions (`fixtures/registry.yaml`); live-hardware checks are additional, marked `hardware`.
- Performance/accuracy log: include relevant latency, failure cases, quality and version numbers.

## Completion checklist
- [ ] API documentation and schema tests complete.
- [ ] Deterministic offline test passes and reproduction command recorded.
- [ ] Integration gate evidence and quantitative metrics saved.
- [ ] Known limitations recorded; no false success or hidden fallback.
- [ ] Handoff report names changed files, tests, residual risks and next phase.
