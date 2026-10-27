# MVStreamRefine V2 image (ADR-0006). Derived from the V1 Dockerfile base so later
# phases (torch==2.5.1+cu121, SAM 2.1) share the same CUDA runtime.
#
#   docker compose build dev        -> mvsr:cu121      (pure CPU core, tests, lint)
#   docker compose build rig        -> mvsr:cu121-rig  (+ pyrealsense2, Robotic_Arm)
#
# Versions are constrained by requirements.lock (pip freeze of the rig image).

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# Host CUDA paths must never leak into this image.
ENV LD_LIBRARY_PATH=

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv git \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade "pip>=24" "setuptools>=69" wheel

WORKDIR /workspace

# Dependencies only (cached independently of source changes): install the project
# from a stub package to resolve [dev] deps, then remove the stub.
FROM base AS dev-deps
COPY pyproject.toml requirements.lock /tmp/mvsr-deps/
RUN mkdir -p /tmp/mvsr-deps/src/mvsr \
 && touch /tmp/mvsr-deps/src/mvsr/__init__.py \
 && python3 -m pip install -c /tmp/mvsr-deps/requirements.lock "/tmp/mvsr-deps[dev]" \
 && python3 -m pip uninstall -y mvsr \
 && rm -rf /tmp/mvsr-deps

FROM dev-deps AS rig-deps
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 udev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/requirements.lock
RUN python3 -m pip install -c /tmp/requirements.lock \
      "pyrealsense2>=2.54,<3" "Robotic_Arm==1.1.6" \
 && python3 -c "import pyrealsense2; from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e; print('rig SDK imports OK')" \
 && rm /tmp/requirements.lock

# Source is bind-mounted over /workspace by compose; the editable install keeps
# `import mvsr` and the `mvsr` console script pointing at /workspace/src.
FROM dev-deps AS dev
COPY . /workspace
RUN python3 -m pip install --no-deps -e /workspace \
 && mvsr --version
CMD ["bash"]

FROM rig-deps AS rig
COPY . /workspace
RUN python3 -m pip install --no-deps -e /workspace \
 && mvsr --version
CMD ["bash"]
