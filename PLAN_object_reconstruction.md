# PLAN — 单腕部相机物体增量重建

## 1. 目标

对机械臂腕部 RealSense 的离线同步 RGB-D 数据进行 object-centric
incremental reconstruction，并让离线与后续实时系统共享核心数据模型和
重建 pipeline。

```text
Wrist RGB-D
    ↓
SAM 2.1 object mask
    ↓
Masked RGB-D
    ↓
Robot pose + hand-eye
    ↓
Keyframe selection
    ↓
Optional local ICP
    ↓
TSDF point cloud / mesh
```

当前系统只有一个逻辑相机，统一命名为 `cam`。历史数据继续使用物理前缀
`wrist`，历史 head 数据保留但不读取。

## 2. 坐标系

内部变换采用 `T_A_B`：将 B 系点变换到 A 系。WORLD 固定为 robot base：

```text
pose source            = T_base_tcp
handeye wrist_cam2     = T_tcp_cam
WORLD                   = robot base
T_world_cam             = T_base_tcp @ T_tcp_cam
Open3D extrinsic        = inv(T_world_cam) = T_cam_world
```

`base_cam1` 不再参与任何计算。Open3D extrinsic 的求逆只允许在
`TSDFVolume` wrapper 内发生。

## 3. 数据约定

```text
data/
├── frame-{idx:06d}_wrist_color.jpg
├── frame-{idx:06d}_wrist_depth.png
├── frame-{idx:06d}_pose.txt
├── JointStates.txt
├── intrinsic/wrist_cam_K.txt
└── handeye/handeye_tf.txt
```

- pose：`x,y,z,qx,qy,qz,qw`，四元数顺序 `xyzw`
- depth：uint16 PNG，单位 mm，`depth.scale = 1000`
- `handeye_tf.txt` 使用 legacy 标签 `wrist_cam2`，语义为 `T_tcp_cam`
- `cam_prefix: wrist` 把物理文件名映射到逻辑 `cam`
- head 图像、head 内参和 `base_cam1` 可留在旧数据集中，但 loader 忽略

## 4. 核心模块

```text
data/models.py
    FramePacket(index, cam, T_world_cam, tcp_pose, raw_pose_7d)

data/offline_source.py
    wrist RGB-D + JointStates + T_tcp_cam → FramePacket

segmentation/
    SAM2Segmenter + MaskCache
    new: output/masks/cam/
    read fallback: output/masks/cam2/

geometry/rgbd.py
    mask erosion + depth range filtering

fusion/keyframe_selector.py
    pose increment + mask quality gates

registration/icp_refiner.py
    robot pose initialization + point-to-plane ICP + mandatory fallback

fusion/tsdf_volume.py
    Open3D VoxelBlockGrid integration / extraction / save

pipeline/offline_pipeline.py
    single cam keyframe → optional ICP → TSDF

visualization/live_viewer.py
    robot-base frame + current cam frame + cam trajectory + model
```

## 5. Pipeline

对每个 `FramePacket`：

1. 从 mask cache 读取物体 mask。
2. 对 RGB-D 做 mask erosion、无效深度清理和深度范围过滤。
3. 用 `T_world_cam`、mask area、valid depth ratio 做关键帧准入。
4. 若 ICP 关闭，直接使用机器人位姿。
5. 若 ICP 开启，以当前帧 cam 系点云为 source、积分当前帧之前的 WORLD
   模型为 target、机器人位姿为初值。
6. ICP 只有在 fitness、RMSE、平移修正量和旋转修正量全部过门限时才接受；
   否则回退机器人位姿。
7. 将关键帧积分进 WORLD=robot base 的 TSDF。
8. 定期保存点云快照、更新 viewer 并写 debug 记录。

## 6. 配置基线

```yaml
dataset:
  cam_prefix: wrist
  pose_semantics: T_base_tcp
  quaternion_order: xyzw
  handeye_convention: wrist_cam2=T_tcp_cam

world_frame: robot_base

keyframe:
  translation_m: 0.02
  rotation_deg: 5.0
  min_mask_area_px: 1000
  min_valid_depth_ratio: 0.7

tsdf:
  device: CPU:0
  voxel_size_m: 0.002
  trunc_voxel_multiplier: 4.0

icp:
  enabled: false
  max_correspondence_distance_m: 0.01
  min_fitness: 0.4
  max_rmse_m: 0.008
  max_translation_correction_m: 0.02
  max_rotation_correction_deg: 5.0
```

## 7. 验收顺序

### Phase 0 — 数据

`tools/inspect_dataset.py`

- wrist RGB/depth 完整
- wrist 内参与图像尺寸一致
- pose 表与逐帧 pose 一致
- `wrist_cam2` 是合法刚体矩阵
- head 文件是否存在不影响结果

### Phase 1 — 坐标链

`tools/validate_transforms.py`

- cam 随机械臂运动时，静态场景映射到 base 后保持稳定
- confirmed `T_tcp_cam` 必须优于 `inv(T_tcp_cam)`
- 不再使用跨相机 overlap

### Phase 2 — Mask

`tools/precompute_masks.py`

- 单路 SAM tracking 覆盖全部帧
- mask 无明显背景污染或跟踪漂移
- 新缓存写到 `output/masks/cam`
- 重建可读取 legacy `output/masks/cam2`

### Phase 3 — TSDF smoke

`tools/validate_tsdf_cam.py`

- 使用真实 `T_world_cam` 积分多个关键帧
- 点云非空、尺寸合理、无坐标轴翻转

### Phase 4 — 完整离线重建

`tools/run_offline_reconstruction.py`

- 至少一个关键帧被积分
- cam 运动时模型覆盖逐渐增大
- 静态物体不随相机产生明显整体漂移
- 无明显双层表面、尺寸错误或背景泄漏
- 输出 point cloud、mesh、snapshot、debug jsonl 和日志

### Phase 5 — ICP A/B

```text
--icp false → output/run_robot_pose/
--icp true  → output/run_icp_pose/
```

比较表面厚度、重影、边缘锐度和轨迹一致性。ICP 失败必须回退 robot pose；
在 A/B 证明改善之前保持默认关闭。

## 8. 离线 MVP 完成定义

- [x] 单 wrist RGB-D 与机械臂 pose 可完整读取
- [x] WORLD 定义为 robot base
- [x] `T_world_cam = T_base_tcp @ T_tcp_cam`
- [x] cam1/head 不参与 loader、pipeline 或验收
- [x] 单路 mask 预计算与 legacy cam2 cache 读取兼容
- [x] 单 cam keyframe TSDF pipeline
- [x] ICP 可关闭且失败强制回退
- [x] point cloud / mesh / debug / logging
- [ ] 目标 Docker 中重新完成 transform、TSDF 与完整重建 smoke test
- [x] 接入实时 FrameSource（代码完成，待硬件联调）
- [ ] 性能优化（thread / queue / GPU / frame dropping）

## 8.1 实时模块（已实现，待硬件验收）

复用离线核心，新增：

```text
data/realsense_source.py        RealSense bgr8+z16 对齐采集，depth_scale 与
                                depth.scale 一致性 fail-fast，host monotonic 时间戳
data/robot_pose_source.py       RealMan TCP 位姿后台轮询线程（xyzrpy → T_base_tcp），
                                带时间戳环形缓冲与 pose_at() 就近查询
data/realtime_source.py         相机帧 + 就近机械臂位姿 → FramePacket；
                                超过 sync.max_error_ms 的帧丢弃计数
segmentation/sam2_stream_segmenter.py  SAM 2.1 StreamTracker 逐帧跟踪适配
pipeline/reconstruction_engine.py      离线/实时共享的逐帧重建核心
                                       （preprocess → keyframe → ICP → TSDF）
pipeline/realtime_pipeline.py   状态机 PREVIEW / MASK_CONFIRM / RUNNING /
                                PAUSED / LOST（UI 无关，可单测）
tools/run_realtime_reconstruction.py   OpenCV 交互 + Open3D 增量 viewer
configs/realtime.yaml           相机 SN、机械臂 IP、同步与跟踪阈值
```

硬件验收清单：

- [ ] RealSense 出流、depth_scale 校验通过、内参合理
- [ ] 机械臂位姿轮询稳定（read_failures 不增长），同步误差 < 50ms
- [ ] ROI → SAM 跟踪 → 确认 → 积分全流程可用，帧率可接受
- [ ] LOST 恢复与 N（新模型）行为正确
- [ ] 导出的实时点云与离线结果量级一致

## 9. 后续路线

```text
Offline known-pose TSDF
        ↓
Stable local ICP
        ↓
Realtime FrameSource
        ↓
Pose graph / reintegration
        ↓
Coverage / uncertainty
        ↓
Next-best-view / robot motion
```

在单相机 TSDF baseline 稳定之前，不增加全局配准、pose graph、NeRF、
BundleSDF 或主动感知。
