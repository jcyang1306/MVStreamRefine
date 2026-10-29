# ADR-0006: Repository layout, toolchain and runtime environment
Status: Accepted (owner confirmed 2026-10-27). Date: 2026-10-27.

## Machines
- **Workstation `CW20175`** (user `hkclr_user`, LAN IP `192.168.2.63`; the hostname does not resolve from the laptop, use the IP or an `~/.ssh/config` alias): RTX 3060 12 GB, NVIDIA driver 570.133.07 (CUDA ≤ 12.8; cu121 wheels/images are compatible), Docker Compose + NVIDIA Container Toolkit verified 2026-10-27. Wrist D405 and RM65 are connected here.
- **All development, tests, recording and GPU work happen on the workstation**, edited through Cursor Remote-SSH. The laptop is only an SSH client.

## Repository
- V2 repository root on the workstation: `~/projects/chuang/MVStreamRefine` (new git repo). Planning docs stay at their current relative paths (`MASTER_PLAN.md`, `contracts/`, `subplans/`, `decisions/`). Docs and code use **repo-relative paths only**.
- **V1 reference location convention:** `ref/MVStreamRefine` inside the V2 repo, either the V1 checkout itself or a symlink to it on the workstation. Override with `MVSR_V1_ROOT`. Read-only, git-ignored, mounted read-only into containers. V2 never imports V1 modules; algorithms are ported through explicit adapters with parity tests.
- V1 data is not reused (ADR-0007); nothing reads V1 paths at run time. The reference hand-eye file is copied into each session at record time.
- Python package `src/mvsr/` (src layout); tests in `tests/`; CLI `mvsr`.
- Recorded and imported data: `./data` (`$MVSR_DATA_ROOT`), git-ignored; sessions under `./data/sessions/<session_id>/`. Budget ≈ 6 GB per minute of 30 fps lossless RGB-D.

## Runtime
- Docker only; no host Python environment is maintained.
- One image `mvsr:cu121`, derived from the V1 Dockerfile (`nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`, system Python 3.10). Compose services:
  - `dev`: source mounted read-write; runs `pytest`, `ruff`, `mypy`, `mvsr` CLI; no USB/X11.
  - `rig`: overlay adding `privileged`, `/dev/bus/usb`, host network (RM65 at `192.168.1.19`), X11 for previews.
- Containers run as the host user (`user: "${UID}:${GID}"`) so outputs are not root-owned.
- Mounts: repo (rw), `./data` (rw), `ref/MVStreamRefine` resolved target (ro).
- Hardware tests are `@pytest.mark.hardware` and run only in `rig`.

## Toolchain
- Python 3.10 (image system Python).
- Lint/format `ruff`; type check `mypy --strict` on `mvsr.core`; tests `pytest` + `hypothesis` for SE(3) property tests.
- Internal types: frozen `dataclasses` with explicit validators (no pydantic). JSON metadata via stdlib; arrays via NPZ; images via PNG (`opencv-python-headless`).
- P0 runtime deps: `numpy`, `pyyaml`, `opencv-python` (PNG IO and the optional preview window), `pyrealsense2>=2.54,<3`, `Robotic_Arm==1.1.6` (recorder, ADR-0007); `scipy` test-only. Later phases add `open3d`, `torch==2.5.1+cu121`, `sam2-inference` (vendored from V1). All pinned in `requirements.lock` after the first green build.
- Workstation OS is Ubuntu 20.04; everything runs in the Ubuntu 22.04 image, so host Python/library versions do not matter.

## Amendment (2026-10-29, P2)
- The Dockerfile has a `torch-deps` stage below `dev-deps`: `torch==2.5.1` / `torchvision==0.20.1` from the cu121 index, then `third_party/sam2-inference` (vendored byte-for-byte from V1 `src/segmention/sam2`, V1 commit `10006e3`, Apache-2.0; inference only). Both `dev` and `rig` images contain it (≈ 9.4 GB).
- The GPU is reached through the default `nvidia` Docker runtime (`NVIDIA_VISIBLE_DEVICES=all` on the `dev` service); no compose device reservation is needed on CW20175.
- SAM 2.1 weights are not in git: `checkpoints/sam2.1_hiera_tiny.pt` (copied from V1 `checkpoints/`, git-ignored) with sha256 `7402e0d8…be69` pinned in `configs/observation.yaml`; a mismatch refuses segmentation.
- `torch` / `sam2_inference` are imported lazily; tests that need them skip without a GPU or checkpoint.
