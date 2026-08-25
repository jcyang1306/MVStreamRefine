# MVStreamRefine

机械臂腕部单 RealSense 的 object-centric 增量 RGB-D 重建。

- 唯一逻辑相机名：`cam`
- 物理数据前缀：`wrist`（保留旧数据命名）
- 世界坐标系：robot base
- 位姿链：`T_world_cam = T_base_tcp @ T_tcp_cam`
- 主链路：SAM 2.1 mask → masked RGB-D → keyframe → optional ICP → TSDF

## 数据约定

```text
data/
├── frame-XXXXXX_wrist_color.jpg
├── frame-XXXXXX_wrist_depth.png
├── frame-XXXXXX_pose.txt
├── JointStates.txt
├── intrinsic/wrist_cam_K.txt
└── handeye/handeye_tf.txt
```

- 7D pose：`x,y,z,qx,qy,qz,qw`，语义为 `T_base_tcp`
- hand-eye 继续使用物理标签 `wrist_cam2`，语义为 `T_tcp_cam`
- depth：uint16 PNG，单位 mm，`depth.scale = 1000`
- legacy head 文件、head 内参和 `base_cam1` 标定块允许留在数据目录，但不会被读取

## 本地开发

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
python tools/inspect_dataset.py --config configs/offline.yaml
```

## Docker

目标环境：RTX 3060、NVIDIA driver 535；镜像使用 CUDA 12.1 runtime。

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
docker compose build
docker compose run --rm reconstruction \
  python3 tools/inspect_dataset.py --config configs/offline.yaml
```

SAM 2.1 checkpoint 放到 `checkpoints/`，容器内挂载为
`/models/sam2.1_hiera_tiny.pt`。

## 1. Mask 预计算

交互选择首帧 ROI：

```bash
xhost +si:localuser:root
docker compose run --rm \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  reconstruction \
  python3 tools/precompute_masks.py --config configs/offline.yaml
xhost -si:localuser:root
```

无窗口时传 `xyxy`：

```bash
docker compose run --rm reconstruction \
  python3 tools/precompute_masks.py --config configs/offline.yaml \
  --cam-box X1 Y1 X2 Y2
```

新结果写入 `output/masks/cam/*.png`。重建也兼容读取已有的
`output/masks/cam2/*.png`，但不再读取 cam1 mask。

## 2. 坐标变换验证

验证腕部相机随机器人运动时，静态场景在 robot base 下保持一致，并与错误
的 `inv(T_tcp_cam)` 方向对照：

```bash
docker compose run --rm reconstruction \
  python3 tools/validate_transforms.py --config configs/offline.yaml
```

报告写入 `output/debug/transform_validation.json`，可视化点云与轨迹写入
`output/debug/transforms/`。该工具不再计算跨相机 overlap。

## 3. 单相机 TSDF smoke test

使用真实 `T_world_cam` 积分前若干个关键帧：

```bash
docker compose run --rm reconstruction \
  python3 tools/validate_tsdf_cam.py \
  --config configs/offline.yaml --keyframes 5
```

结果保存为 `output/debug/tsdf_cam.ply`。

## 4. 完整离线重建

无窗口运行：

```bash
docker compose run --rm reconstruction \
  python3 tools/run_offline_reconstruction.py \
  --config configs/offline.yaml --no-viewer
```

带 X11 viewer 时去掉 `--no-viewer`。窗口显示 robot-base 原点、当前 cam
坐标系、cam 轨迹和 TSDF 点云；热键：SPACE 暂停、S 保存点云、M 保存
mesh、Q 提前退出。

输出：

```text
output/pointcloud/object_a.ply
output/mesh/object_a_mesh.ply
output/pointcloud/model_kf_XXX.ply
output/debug/fusion_debug.jsonl
output/logs/run_*.log
```

debug 位姿字段为 `T_world_cam` 和 `T_world_cam_used`，WORLD 均指 robot base。

## 5. ICP A/B

ICP 默认关闭。它只允许在 robot pose 附近做局部微调；fitness、RMSE 或
修正量门限不通过时必须回退 `T_world_cam`。

```bash
docker compose run --rm reconstruction \
  python3 tools/run_offline_reconstruction.py \
  --config configs/offline.yaml --no-viewer --icp false

docker compose run --rm reconstruction \
  python3 tools/run_offline_reconstruction.py \
  --config configs/offline.yaml --no-viewer --icp true
```

结果分别写入 `output/run_robot_pose/` 与 `output/run_icp_pose/`。重点比较
单相机多帧融合后的表面厚度、重影、边缘锐度和轨迹一致性；确认 ICP 改善
后才应将 `icp.enabled` 设为 true。
