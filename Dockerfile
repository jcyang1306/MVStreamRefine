FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
# Host CUDA paths must never leak into this image (cu121 wheels ship their own libs).
ENV LD_LIBRARY_PATH=

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      libgomp1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch must be installed before sam2-inference and pinned to cu121 (driver 535).
COPY src/segmention/sam2 /tmp/sam2-inference
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install /tmp/sam2-inference \
 && python3 -c "import sam2, sam2_inference; print('sam2-inference import OK')"

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install -r requirements.txt

# Realtime hardware adapters. Pin the RealMan SDK so image rebuilds do not
# silently change the robot API; pyrealsense2 provides the RealSense runtime.
RUN python3 -m pip install \
      "pyrealsense2>=2.54,<3" \
      "Robotic_Arm==1.1.6" \
 && python3 -c "import pyrealsense2; from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e; print('realtime hardware SDK imports OK')"

COPY . /workspace
RUN python3 -m pip install --no-deps -e .

CMD ["python3", "tools/run_offline_reconstruction.py", "--config", "configs/offline.yaml"]
