FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/sam2-inference
COPY . .

# CUDA 12.1 wheels run with NVIDIA driver 535. torch stays a runtime
# dependency and is deliberately absent from the isolated build requirements.
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install . \
    && python3 -c "import sam2, sam2_inference; print('sam2-inference import OK')"

CMD ["python3", "-c", "import torch; print('CUDA available:', torch.cuda.is_available())"]
