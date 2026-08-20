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

## Mask 预计算（Task 3，需在 CUDA 容器内运行）

不传 `--cam1-box` / `--cam2-box` 时，工具会依次显示 head 和 wrist
首帧。鼠标拖框后按 Enter/Space 确认（按 C 取消）。Docker 需要传入宿主机
X11 显示：

```bash
xhost +si:localuser:root
docker compose run --rm \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  reconstruction \
  python3 tools/precompute_masks.py --config configs/offline.yaml
xhost -si:localuser:root
```

无图形界面或自动化运行时仍可显式传入像素坐标 `xyxy`，此时不会打开窗口：

```bash
docker compose run --rm reconstruction \
  python3 tools/precompute_masks.py --config configs/offline.yaml \
  --cam1-box X1 Y1 X2 Y2 --cam2-box X1 Y1 X2 Y2
```

输出：`output/masks/cam{1,2}/*.png`（0/255）、`mask_metadata.jsonl`
（mask_area / iou_prev / valid_depth_ratio）以及 `output/masks/previews/`
叠加图供人工抽检。重建阶段默认读取该缓存，不重复执行 SAM 2.1。

## TSDF cam1 单视角验收（Task 5，需在 CUDA 容器内运行）

依赖 Task 3 的 mask 缓存（`output/masks/`）。cam1（head）固定为 WORLD，
取前 N 帧（默认 10）以单位位姿积分进 TSDF，导出点云：

```bash
docker compose run --rm reconstruction \
  python3 tools/validate_tsdf_cam1.py --config configs/offline.yaml
```

成功时打印每帧 mask 面积 / 有效深度比、总点数与边界盒，并保存
`output/debug/tsdf_cam1.ply`（可用 MeshLab / CloudCompare 检查物体形状）。
点数为 0 或 mask 缓存缺失时以非零退出码失败。帧数可用 `--frames` 调整。

## 双相机增量重建（Task 6–7，需在 CUDA 容器内运行）

完整融合循环（无 ICP）：cam1 低频锚定（前 `cam1.initial_frames` 帧 +
每 `update_interval_frames` 一帧），cam2 按关键帧准入（位移 > 2cm 或
旋转 > 5°，且 mask 面积 / 有效深度比过门限）以已知位姿 `T_world_cam2` 积分：

```bash
docker compose run --rm reconstruction \
  python3 tools/run_offline_reconstruction.py --config configs/offline.yaml
```

成功时打印 cam1 积分次数、cam2 关键帧数（及被拒帧数）、总点数与边界盒，
并保存 `output/pointcloud/object_a.ply`。验收现象：随 cam2 运动，物体模型
比单视角更完整（侧面/背面补全）；无任何 cam2 关键帧时以非零退出码失败。
调试可加 `--max-frames N`。

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
