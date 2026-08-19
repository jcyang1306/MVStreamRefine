# MVStreamRefine

双 RealSense（head 固定 + wrist 眼在手上）离线 RGB-D 数据的 object-centric 增量重建。
完整开发计划见 `PLAN_object_reconstruction.md`。

## 仓库结构

```text
data/                      离线采集数据（113 帧 head/wrist RGB-D + pose + 标定）
src/segmention/sam2/       自包含 SAM 2.1 推理包（sam2-inference）
src/object_reconstruction/ 重建工程（按 PLAN Task 顺序实现中）
tools/                     CLI 工具
configs/offline.yaml       离线 pipeline 配置
Dockerfile / compose.yaml  部署镜像（CUDA 12.1 runtime，RTX 3060 + driver 535）
```

## 本地开发（Task 1 数据检查）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python tools/inspect_dataset.py --config configs/offline.yaml
pytest
```

## Docker 部署（目标机：RTX 3060，driver 535）

```bash
# 先验证 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

docker compose build
docker compose run --rm reconstruction \
  python3 tools/inspect_dataset.py --config configs/offline.yaml

# GPU 通过 compose.yaml 的 deploy.resources.reservations.devices 声明。
# 若 compose 版本过旧仍报 schema 错误，可退回：
# docker build -t mvstreamrefine:cu121 .
# docker run --rm --gpus all --shm-size=8g ... mvstreamrefine:cu121
```

SAM 2.1 checkpoint 不入库，放在 `checkpoints/` 并通过只读 volume 挂载为
`/models/sam2.1_hiera_tiny.pt`。

## 数据语义（已确认，2026-08-19）

- `pose_semantics = T_base_tcp`：7D 位姿为 `x,y,z,qx,qy,qz,qw`
- `quaternion_order = xyzw`
- `handeye`：`wrist_cam2 = T_tcp_cam2`，`base_cam1 = T_base_cam1`
- `depth.scale = 1000.0`（采集端将米 ×1000 存为 uint16 mm）

`T_world_cam2 = inv(T_base_cam1) @ T_base_tcp @ T_tcp_cam2`（WORLD = cam1/head）。

## 坐标变换验证状态

`tools/validate_transforms.py` 实测通过（2026-08-19，重标定 `base_cam1` 后）：

- cam2 运动时静止场景世界点云跨帧一致性 0.71（翻转 wrist 手眼后降至 0.44）；
- cam2 点云与固定 cam1 点云跨相机重叠 0.34（所有翻转变体仅 0.01–0.02）。

当前无阻塞项，双相机融合链路可用。
