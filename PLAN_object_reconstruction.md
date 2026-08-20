# PLAN.md — 双 RealSense 物体 A 离线增量重建 → 实时重建

## 0. 目标

实现一个 Python 工程，对已经离线采集并同步的数据进行 **object-centric incremental RGB-D reconstruction**。

已知条件：

- `RealSense1 = head = cam1`：固定相机，定义世界坐标系 `WORLD = head optical frame`。
- `RealSense2 = wrist = cam2`：眼在手上，随机械臂运动。
- 对每个时间点能够拿到同步数据：
  - `rgb1`
  - `depth1`
  - `rgb2`
  - `depth2`
  - `T_world_cam2`，或能够由机器人 TCP pose + 手眼标定计算得到它。
- `RealSense1` 的位姿恒定：
  - `T_world_cam1 = I`
- 场景中央存在静态物体 A。
- 两个相机都可以观察到物体 A。
- 使用仓库内 `src/segmention/sam2` 提供的 SAM 2.1 推理包得到物体 A 的 mask。
- 最终目标：
  1. 离线数据首先跑通；
  2. RealSense2 运动过程中，物体点云逐渐完整；
  3. 输出 point cloud / mesh；
  4. 后续接 RealSense 实时模块时，不重写核心 reconstruction pipeline；
  5. 后续可增加 ICP / pose graph / BundleSDF / active perception。

---

# 1. 核心设计原则

## 1.1 第一版不要直接实现 NeRF / BundleSDF

第一版主链路固定为：

```text
Offline RGB-D
    ↓
SAM 2.1 mask
    ↓
Masked RGB-D
    ↓
Known camera pose
    ↓
TSDF Fusion
    ↓
Incremental Point Cloud / Mesh
```

必须先证明：

1. 双相机坐标变换正确；
2. depth 单位正确；
3. RGB-depth 对齐正确；
4. mask 正确；
5. 不做 ICP 时，已知机器人 pose 已经可以完成基本融合。

只有 baseline 正确后，再增加：

```text
Robot pose
    ↓
ICP refinement
    ↓
Refined camera pose
    ↓
TSDF
```

再之后才考虑：

```text
Pose Graph
Bundle Adjustment-like optimization
Neural SDF / BundleSDF
```

---

# 2. 坐标系约定

整个工程只能使用一种 pose 命名规则：

```text
T_A_B
```

含义：

> 将 B 坐标系中的三维点变换到 A 坐标系。

即：

```python
p_A = T_A_B @ p_B
```

统一：

```text
WORLD = head optical frame
C1    = head optical frame（RealSense1）
C2    = wrist optical frame（RealSense2）
```

因此：

```text
T_world_cam1 = I
T_world_cam2 = 当前 RealSense2 camera-to-world pose
```

如果输入是：

```text
T_cam2_world
```

必须在 DataSource 层转换一次：

```python
T_world_cam2 = np.linalg.inv(T_cam2_world)
```

之后 reconstruction 内部禁止混用 pose convention。

Open3D TSDF `extrinsic` 使用时，封装在 `TSDFVolume` 内部统一转换：

```python
T_cam_world = np.linalg.inv(T_world_cam)
```

上层 pipeline 永远只传 `T_world_cam`。

---

# 3. 工程目录

建议 Cursor 按下面结构创建：

```text
object_reconstruction/
├── README.md
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── offline.yaml
│   └── realtime.yaml
│
├── src/
│   └── object_reconstruction/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── models.py
│       │   ├── frame_source.py
│       │   ├── offline_source.py
│       │   └── realsense_source.py
│       │
│       ├── calibration/
│       │   ├── intrinsics.py
│       │   └── transforms.py
│       │
│       ├── segmentation/
│       │   ├── base.py
│       │   ├── sam2_segmenter.py
│       │   └── mask_cache.py
│       │
│       ├── geometry/
│       │   ├── rgbd.py
│       │   ├── pointcloud.py
│       │   └── pose_utils.py
│       │
│       ├── fusion/
│       │   ├── tsdf_volume.py
│       │   └── keyframe_selector.py
│       │
│       ├── registration/
│       │   └── icp_refiner.py
│       │
│       ├── visualization/
│       │   └── live_viewer.py
│       │
│       ├── pipeline/
│       │   ├── offline_pipeline.py
│       │   └── reconstruction_pipeline.py
│       │
│       └── utils/
│           ├── config.py
│           ├── logging.py
│           └── io.py
│
├── tools/
│   ├── inspect_dataset.py
│   ├── normalize_dataset.py
│   ├── precompute_masks.py
│   ├── validate_transforms.py
│   ├── run_offline_reconstruction.py
│   ├── export_model.py
│   └── run_realtime_reconstruction.py
│
├── tests/
│   ├── test_transforms.py
│   ├── test_projection.py
│   ├── test_offline_source.py
│   └── test_keyframe_selector.py
│
└── output/
    ├── masks/
    ├── debug/
    ├── pointcloud/
    ├── mesh/
    └── logs/
```

## 3.1 当前仓库模块边界

当前仓库已经包含可独立安装的 SAM 2.1 最小推理包：

```text
src/segmention/sam2/
├── src/sam2/
├── src/sam2_inference/
├── examples/
├── tests/
├── pyproject.toml
└── Dockerfile
```

重建工程只依赖其稳定公开接口：

```python
from sam2_inference import ImageSegmenter, VideoTracker, StreamTracker
```

禁止从 reconstruction 代码直接调用 `sam2` 内部 predictor 的私有方法。也不要同时安装官方
`SAM-2` wheel 和本仓库的 `sam2-inference`，两者会占用同一个 `sam2` Python 模块名。

除上述 SAM 2.1 子包外，本节规划的 `object_reconstruction/`、`configs/`、`tools/`、
mask cache 和 TSDF 主工程当前均尚未实现，仍属于后续 Task 的目标结构。仓库中也不包含
SAM 2.1 `.pt/.pth` 权重；checkpoint 必须作为外部只读文件挂载，禁止打进应用镜像或提交
到 Git。

当前 `src/` 尚未被 Git 跟踪。开始实现 Task 1 前必须明确把 SAM 2.1 子包作为普通源码
纳入版本管理，或固定为带 commit SHA 的 submodule；否则 Docker/CI 无法从干净 checkout
复现。不要依赖开发机上未跟踪的目录。

## 3.2 Docker / GPU 环境基线（RTX 3060 + NVIDIA 535）

宿主机 `nvidia-smi` 显示的 `CUDA Version: 12.2` 表示 **535 驱动可支持的最高 CUDA
driver API**，不表示容器必须安装 CUDA 12.2 toolkit。当前 SAM 2.1 子包的 Dockerfile
采用以下固定组合；根工程镜像仍待实现和整体验证：

```text
container CUDA runtime : 12.1.1 + cuDNN 8
PyTorch wheel          : torch 2.5.1 + cu121
torchvision wheel      : torchvision 0.20.1 + cu121
Python                 : 3.10（Ubuntu 22.04 默认）
host NVIDIA driver     : 535（满足 CUDA 12.1 runtime 要求）
GPU                    : RTX 3060 / compute capability 8.6
```

因此容器统一使用 CUDA 12.1，而不是混装系统 CUDA 12.2、pip CUDA 库和其他版本的
PyTorch。宿主机只需要：

```text
NVIDIA driver 535
Docker Engine
NVIDIA Container Toolkit
```

Task 1 创建根 `Dockerfile`、`requirements.txt` 和 Python 包之后，按下面的目标模板从
仓库根目录构建主镜像；在这些文件落地前，根目录的 `docker build` 命令不可运行：

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY src/segmention/sam2 /tmp/sam2-inference
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install /tmp/sam2-inference

COPY . /workspace
# requirements.txt 不得再次声明 torch/torchvision 或覆盖 cu121 wheel。
RUN python3 -m pip install -r requirements.txt \
 && python3 -m pip install -e .

CMD ["python3", "tools/run_offline_reconstruction.py", "--config", "configs/offline.yaml"]
```

`requirements.txt` 至少覆盖 reconstruction 自身的 NumPy、Pillow、PyYAML、OpenCV 和
Open3D 依赖；在完成容器 smoke test 后生成锁定文件，禁止部署时无版本地安装“最新版”。
SAM 2.1 包自身的最低依赖以 `src/segmention/sam2/pyproject.toml` 为准。

运行示例：

```bash
docker build -t mvstreamrefine:cu121 .
docker run --rm --gpus all \
  --shm-size=8g \
  -e SAM2_CHECKPOINT=/models/sam2.1_hiera_tiny.pt \
  -v "$PWD/data:/workspace/data:ro" \
  -v "$PWD/checkpoints:/models:ro" \
  -v "$PWD/output:/workspace/output" \
  mvstreamrefine:cu121
```

对应的目标 `compose.yaml`：

```yaml
services:
  reconstruction:
    build:
      context: .
      dockerfile: Dockerfile
    image: mvstreamrefine:cu121
    shm_size: "8gb"
    environment:
      SAM2_CHECKPOINT: /models/sam2.1_hiera_tiny.pt
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./data:/workspace/data:ro
      - ./checkpoints:/models:ro
      - ./output:/workspace/output
```

不要使用服务级 `gpus: all`：部分 Compose schema 会报 `Additional property gpus is not allowed`。
GPU 通过 NVIDIA Container Toolkit 的 `deploy.resources.reservations.devices` 声明。
构建主镜像前先独立验证宿主机 NVIDIA Container Toolkit：

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

容器启动 smoke test：

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import sam2, sam2_inference; print('SAM2 import OK')"
```

预期 `torch.version.cuda == "12.1"` 且 `torch.cuda.is_available() is True`。若后者为
False，先检查 NVIDIA Container Toolkit 和 `--gpus all`，不要在容器里安装宿主机驱动。
不要把宿主机 `/usr/local/cuda-12.2` 挂入容器，也不要显式透传指向宿主机 CUDA 的
`LD_LIBRARY_PATH`；容器只使用镜像和 cu121 wheel 内的库。

安装 Open3D 后还要单独验证其设备能力：

```bash
python3 - <<'PY'
import open3d as o3d
print("open3d", o3d.__version__)
print("open3d CUDA available", o3d.core.cuda.is_available())
grid = o3d.t.geometry.VoxelBlockGrid(
    attr_names=("tsdf", "weight", "color"),
    attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.uint8),
    attr_channels=((1,), (1,), (3,)),
    voxel_size=0.01,
    block_resolution=16,
    block_count=100,
    device=o3d.core.Device("CPU:0"),
)
print("Open3D CPU VoxelBlockGrid OK")
PY
```

普通 `pip install` 默认不编译 SAM 2.1 的 connected-components CUDA 扩展，不需要
`nvcc`，只会跳过小孔填充后处理，适合作为第一版。若确认必须启用该扩展，改用
`nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` 构建阶段，并执行：

```bash
SAM2_BUILD_CUDA=1 python3 -m pip install --no-build-isolation /tmp/sam2-inference
```

`open3d` 的普通 pip wheel 是否支持 CUDA必须在镜像内实测，不能因为 PyTorch 可见 GPU
就假定 Open3D 也可见。MVP 默认让 SAM 2.1 使用 GPU、TSDF 使用 `CPU:0`；只有使用
CUDA-enabled Open3D 构建并通过最小积分测试后，才把 TSDF 改为 `CUDA:0`。

RTX 3060 显存有限且存在桌面版/移动版容量差异。部署前执行
`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`。默认选
`sam2.1_hiera_tiny`，离线的 head/wrist 两路顺序运行并在两路之间释放 tracker；
不要默认并发常驻两个 large 模型。

---

# 4. 核心数据模型

在 `src/object_reconstruction/data/models.py` 定义 dataclass。仓库根 `data/` 只存放
采集数据，不放 Python 源码。

```python
@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        ...

@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    mask: np.ndarray | None = None

@dataclass
class FramePacket:
    index: int
    timestamp: float | None

    cam1: CameraFrame
    cam2: CameraFrame

    T_world_cam1: np.ndarray
    T_world_cam2: np.ndarray

    tcp_pose: np.ndarray | None = None
    raw_pose_7d: np.ndarray | None = None
```

要求：

```text
RGB:
uint8
H x W x 3
RGB order

Depth:
uint16 mm
或者 float32 meter
但整个项目必须明确 depth_scale

Mask:
bool
H x W

Transform:
float64
4 x 4
```

---

# 5. 离线数据先标准化

不要让 reconstruction pipeline 直接适配当前所有杂乱文件格式。

## 5.1 当前 `data/` 的真实格式

当前数据集共 113 帧（index `000000`～`000112`），每个 index 均完整包含：

```text
data/
├── frame-000000_head_color.jpg    # 1280 x 720, 8-bit JPEG
├── frame-000000_head_depth.png    # 1280 x 720, 16-bit 单通道 PNG
├── frame-000000_wrist_color.jpg
├── frame-000000_wrist_depth.png
├── frame-000000_pose.txt          # 7 个数
├── ...
├── JointStates.txt                # 113 x 7，和逐帧 pose 仅有文本精度差
├── intrinsic/
│   ├── head_cam_K.txt             # 3 x 3
│   └── wrist_cam_K.txt            # 3 x 3
└── handeye/
    └── handeye_tf.txt             # wrist_cam/head_cam 两个 4 x 4 矩阵
```

第一版明确映射：

```text
head  -> cam1 / 固定相机 / WORLD
wrist -> cam2 / 眼在手上相机
```

当前数据没有显式 timestamp，离线同步只能以 frame index 为准，`FramePacket.timestamp`
设为 `None`，不得伪造采样周期。`JointStates.txt` 虽然文件名如此，但内容是 7 维量而
不是常见的多关节角数组；每帧 `frame-xxxxxx_pose.txt` 是它的低精度副本。实现时选
`JointStates.txt` 为数值源、逐帧文件用于完整性交叉检查。

当前内参文件解析结果应作为 loader 单元测试 fixture：

```text
head : fx=908.781982421875, fy=908.7359619140625,
       cx=648.3536376953125, cy=370.1097106933594
wrist: fx=653.2439575195312, fy=652.7537231445312,
       cx=636.3500366210938, cy=358.8265380859375
```

以下语义已由数据采集端确认（2026-08-19），并已写入 `configs/offline.yaml`：

```text
pose 7D           = x, y, z, qx, qy, qz, qw（quaternion 顺序 xyzw）
pose 语义         = T_base_tcp（TCP 在机械臂 base 系下的位姿）
handeye wrist_cam2 = T_tcp_cam2（末端 → RealSense2）
handeye base_cam1  = T_base_cam1（机械臂 base → RealSense1）
depth.scale       = 1000.0（采集端保存时米 ×1000 → uint16 mm）
```

**实测状态（tools/validate_transforms.py，2026-08-19 重标定 base_cam1 后）**：
两条验收全部通过——cam2 世界点云跨帧静止一致性 ~0.71（翻转 wrist 手眼后降至
~0.44）；cam2 与固定 cam1 的跨相机重叠 ~0.34（所有翻转变体仅 0.01–0.02）。
双相机融合链路已解锁。

## 5.2 标准化适配层

先实现：

```text
OfflineFrameSource
```

它负责把现有数据转换成统一的 `FramePacket`。

`OfflineFrameSource` 首先直接支持上述 flat layout，并在内存中完成
`head/wrist -> cam1/cam2` 映射；`normalize_dataset.py` 再按需生成下面的可移植布局，
不要复制一份数据后才允许做 Phase 0。

推荐标准化数据格式：

```text
dataset/
├── calibration/
│   ├── cam1_intrinsics.json
│   ├── cam2_intrinsics.json
│   └── calibration.json
│
├── frames/
│   ├── 000000/
│   │   ├── rgb1.png
│   │   ├── depth1.png
│   │   ├── rgb2.png
│   │   ├── depth2.png
│   │   └── pose.json
│   ├── 000001/
│   └── ...
│
└── manifest.jsonl
```

`pose.json` 推荐：

```json
{
  "timestamp": null,
  "T_world_cam2": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

建议在 manifest / pose metadata 中同时保留：

```json
{
  "source_index": 0,
  "timestamp": null,
  "raw_pose_7d": [0.19252, -0.202075, 0.262422, 0.799332013, 0.108328487, -0.530382538, 0.260821078],
  "raw_pose_convention": "由采集端确认后填写"
}
```

如果原始数据中存储的是：

```text
robot TCP pose
+
T_base_cam1
+
T_tcp_cam2
```

只允许在 `OfflineFrameSource` 或 calibration 模块计算：

```text
T_world_cam2
```

若采集端最终确认 `pose = T_base_tcp`、`head_cam = T_base_head`、
`wrist_cam = T_tcp_wrist`，且 `WORLD = head`，则唯一允许的组合公式是：

```python
T_world_cam2 = np.linalg.inv(T_base_head) @ T_base_tcp @ T_tcp_wrist
```

这只是基于**已确认语义**的公式，不得直接把 `handeye_tf.txt` 中的标签当作上述方向。
若任一矩阵定义相反，应在 calibration loader 边界统一求逆并改名，pipeline 内不做猜测。

重建模块不要知道 robot base / TCP / hand-eye 的存在。

---

# 6. Phase 0 — 数据检查工具

首先实现：

```text
tools/inspect_dataset.py
```

不要做 reconstruction。

检查每一帧：

```text
五类逐帧文件是否均为连续 000000~000112
RGB1 / depth1 shape
RGB2 / depth2 shape
timestamp
depth dtype
depth min/max
valid depth ratio
T_world_cam2（仅 conventions 已确认时）
rotation determinant（同上）
transform bottom row（同上）
translation range（同上）
JointStates 行与逐帧 pose 的一致性
内参矩阵有限、fx/fy > 0、主点在图像范围内
handeye 旋转 determinant 与齐次矩阵末行
```

当 `pose_semantics`、`quaternion_order` 或 `handeye_convention` 仍为 `null` 时，
`inspect_dataset.py` 只能检查 raw pose、内参、图像和 handeye 矩阵本身，并在报告中标记
阻塞项；此时禁止构造或验证 `T_world_cam2`，更不能假装它已满足旋转/平移检查。

同时输出：

```text
output/debug/dataset_report.json
```

必须检查：

```python
assert T.shape == (4, 4)
assert abs(np.linalg.det(R) - 1) < tolerance
assert np.allclose(T[3], [0, 0, 0, 1])
```

并显示若干同步帧：

```text
RGB1 | Depth1
RGB2 | Depth2
```

## 验收标准

- 113 个 frame index 均具有 head/wrist RGB、depth 和 pose；
- RGB/depth 尺寸对应；
- depth 单位明确；
- camera intrinsics 正确；
- pose 没有 NaN；
- RealSense2 轨迹在 Open3D 中看起来符合机械臂实际运动。

---

# 7. Phase 1 — 先不用 SAM 2.1，不用 TSDF，验证坐标变换

这是整个项目最重要的 debug 阶段。

暂时手工指定一个 ROI 或已有 mask。

实现：

```text
depth
  ↓
back projection
  ↓
camera point cloud
  ↓
T_world_cam
  ↓
world point cloud
```

核心接口：

```python
def depth_to_pointcloud(
    depth,
    intrinsics,
    mask=None,
) -> np.ndarray:
    ...
```

以及：

```python
def transform_points(
    points,
    T_dst_src,
) -> np.ndarray:
    ...
```

生成：

```text
cam1 point cloud
cam2 point cloud
```

统一到 WORLD：

```python
pcd1_world = transform(pcd1_cam1, T_world_cam1)

pcd2_world = transform(pcd2_cam2, T_world_cam2)
```

Open3D 同时显示：

```text
cam1 = 一种颜色
cam2 = 另一种颜色
```

然后播放整段离线序列。

## 最重要的验收现象

当 RealSense2 在运动时：

```text
相机坐标系在运动
但物体 A 的 world point cloud 应基本保持静止
```

如果物体随着机械臂明显漂移：

**立即停止后续开发。**

优先检查：

1. `T_world_cam2` 是否方向反了；
2. hand-eye transform 是否方向反了；
3. mm / m 是否混用；
4. RealSense optical frame 和机械臂 camera frame 是否混淆；
5. 当前离线集的 head/wrist/pose frame index 是否错位；实时数据再检查 timestamp；
6. depth 是否和 RGB 对齐；
7. 内参是否对应当前 resolution。

增加一个脚本：

```text
tools/validate_transforms.py
```

用于只测试：

```text
cam1 PCD
cam2 PCD
camera trajectory
```

---

# 8. Phase 2 — SAM 2.1 离线 mask 预计算

SAM 2.1 不要直接耦合在 TSDF 类里。

定义：

```python
class ObjectSegmenter(Protocol):

    def initialize(...):
        ...

    def segment(self, frame) -> np.ndarray:
        ...
```

实现：

```text
SAM2Segmenter
```

`SAM2Segmenter` 是 reconstruction 侧 adapter，内部只调用仓库提供的公开 API：

```python
tracker = VideoTracker(
    checkpoint=config.checkpoint,
    model_type=config.model_type,  # tiny / small / base_plus / large
    device=config.device,          # cuda / cpu / auto
)
tracker.open(numbered_jpeg_dir)
tracker.add_prompt(0, object_id=1, box=box_xyxy)
for result in tracker.track():
    mask = result.masks[0]  # bool, H x W，源图分辨率
tracker.close()
```

当前 `data/` 混放两路 JPEG，不能直接把根目录传给 `VideoTracker.open()`。预计算工具应
分别建立只含连续编号 JPEG 的 staging 目录（可用 symlink，避免重复占用空间）：

```text
output/sam2_frames/head/00000.jpg  -> data/frame-000000_head_color.jpg
output/sam2_frames/head/00001.jpg  -> data/frame-000001_head_color.jpg
output/sam2_frames/wrist/00000.jpg -> data/frame-000000_wrist_color.jpg
...
```

`VideoTracker` 的 JPEG 目录要求纯数字文件名。输出的 `frame_index` 必须通过 staging
manifest 显式映射回原始 frame index。

离线阶段推荐：

```text
第一帧
  ↓
人工 point / box / initial mask prompt
  ↓
初始化物体 A
  ↓
video tracking
  ↓
所有 frame masks
  ↓
disk cache
```

head 和 wrist 各自作为独立 video stream 运行 SAM 2.1 tracking。RTX 3060 上离线默认
顺序执行两路；每一路结束后 `close()`、删除 tracker，并清理不再需要的 CUDA cache。

输出：

```text
output/masks/cam1/000000.png
output/masks/cam1/000001.png
...

output/masks/cam2/000000.png
output/masks/cam2/000001.png
...
```

以及：

```text
mask_metadata.jsonl
```

每帧记录：

```json
{
  "frame": 100,
  "cam": 2,
  "mask_area": 24500,
  "valid_depth_ratio": 0.91
}
```

之后默认：

```text
reconstruction
优先读取 mask cache
```

而不是每次重新执行 SAM 2.1。

prompt 支持正/负 point、`xyxy` box 或初始 bool mask；MVP 使用首帧人工 box，必要时在
漂移帧追加修正 prompt。Video tracking 没有质量分数，`scores is None`，因此不能把
不存在的 SAM score 写进质量门限；使用 mask area、与前一帧 IoU、有效 depth ratio 和
人工抽检作为缓存质量指标。

---

# 9. Phase 3 — Masked RGB-D preprocessing

实现：

```text
geometry/rgbd.py
```

接口：

```python
def preprocess_object_rgbd(
    rgb,
    depth,
    mask,
    config,
):
    ...
```

流程：

```text
SAM mask
   ↓
optional erosion
   ↓
depth range filter
   ↓
invalid depth filter
   ↓
mask depth
   ↓
RGB-D
```

配置：

```yaml
mask:
  erosion_px: 3
  min_area_px: 1000

depth:
  # 当前 16-bit depth PNG 的单位经采集端确认后填写。
  scale: null
  min_m: 0.15
  max_m: 1.2
```

mask 外：

```python
depth[~mask] = 0
```

注意：

**必须确认 depth image 与 mask 所在 RGB frame 已经完成 pixel alignment。**

否则禁止开始 TSDF。

---

# 10. Phase 4 — TSDF baseline

使用：

```text
open3d.t.geometry.VoxelBlockGrid
```

封装：

```python
class TSDFVolume:

    def integrate(
        self,
        rgb,
        depth,
        intrinsics,
        T_world_cam,
    ):
        ...

    def extract_point_cloud(self):
        ...

    def extract_mesh(self):
        ...

    def save(self, path):
        ...
```

内部统一处理：

```python
T_cam_world = np.linalg.inv(T_world_cam)
```

不要在 pipeline 中到处 `inv()`。

初始配置：

```yaml
tsdf:
  # 普通 Open3D pip wheel 的 MVP 安全默认值。
  device: "CPU:0"
  voxel_size_m: 0.002
  block_resolution: 16
  block_count: 50000
  trunc_voxel_multiplier: 4.0
  depth_max_m: 1.2
  weight_threshold: 3.0
```

`block_count: 50000` 只作为 CPU baseline 上限，运行时必须监控宿主机内存。若目标镜像
已验证 Open3D CUDA，在 RTX 3060 上先从 `block_count: 10000` 开始，根据实际占用逐步
提高，且不要与 SAM 2.1 mask 预计算并发。

只有 CUDA-enabled Open3D 已在目标镜像中通过 smoke test 时才覆盖：

```text
tsdf.device = CUDA:0
```

PyTorch 的 `torch.cuda.is_available()` 不能作为 Open3D CUDA 可用性的证据；启动时必须
分别探测并在请求了 `CUDA:0` 但 Open3D 不支持时 fail fast，不要静默改变计算设备。

---

# 11. Phase 5 — 第一版融合策略

先不要 ICP。

流程：

```python
for packet in source:

    mask1 = mask_cache.get_cam1(packet.index)
    mask2 = mask_cache.get_cam2(packet.index)

    rgbd1 = preprocess(...)
    rgbd2 = preprocess(...)

    if should_integrate_cam1(packet):
        tsdf.integrate(
            rgbd1,
            T_world_cam=np.eye(4)
        )

    if keyframe_selector.accept(packet.T_world_cam2):
        tsdf.integrate(
            rgbd2,
            T_world_cam=packet.T_world_cam2
        )

    if frame_index % visualization_interval == 0:
        model = tsdf.extract_point_cloud()
        viewer.update(model)
```

---

# 12. RealSense1 融合策略

RealSense1 是固定视角。

不要每一帧都和 RealSense2 等权重积分。

第一版：

```yaml
cam1:
  initial_frames: 10
  update_interval_frames: 30
  max_integrations: 30
```

策略：

```text
启动：
cam1 前 10 个有效 frame integration

之后：
低频 integration
```

后续可以进一步改成：

```text
多帧 depth median
→ 一张低噪声 anchor depth
→ integrate
```

---

# 13. Phase 6 — RealSense2 KeyframeSelector

创建：

```python
class KeyframeSelector:

    def should_add(
        self,
        T_world_cam,
        mask,
        depth,
    ) -> bool:
        ...
```

初始条件：

```text
translation > 0.02 m
OR
rotation > 5 degree
```

同时要求：

```text
mask_area > threshold
valid_mask_depth_ratio > threshold
```

配置：

```yaml
keyframe:
  translation_m: 0.02
  rotation_deg: 5.0
  min_mask_area_px: 1000
  min_valid_depth_ratio: 0.7
```

位姿差：

```python
T_delta = inv(T_last) @ T_current
```

translation：

```python
np.linalg.norm(T_delta[:3, 3])
```

rotation：

```python
angle = acos((trace(R) - 1) / 2)
```

必须做数值 clamp。

---

# 14. Phase 7 — 增量可视化

实现：

```text
LiveViewer
```

第一版只显示：

```text
TSDF extracted object point cloud
camera1 coordinate frame
camera2 current coordinate frame
camera2 trajectory
```

每隔：

```text
5~10 个 keyframes
```

执行：

```python
extract_point_cloud()
```

不要每输入一帧就 extract mesh。

增加快捷键：

```text
SPACE : pause
S     : save point cloud
M     : save mesh
Q     : quit
```

保存：

```text
output/pointcloud/model_xxxx.ply
output/mesh/model_xxxx.ply
```

---

# 15. Phase 8 — TSDF baseline 验收

完成到这里时必须得到：

```text
RealSense2 从不同角度观察物体
              ↓
模型覆盖范围不断增大
              ↓
点云逐渐变完整
```

必须保存 debug 数据：

```text
frame id
keyframe id
T_world_cam2
mask area
valid depth ratio
number of TSDF blocks
point cloud point count
```

## Baseline 验收标准

### 几何

- 物体不会随着 RealSense2 运动发生明显整体漂移；
- cam1 / cam2 重叠区域基本重合；
- 不存在明显 2x 尺寸错误；
- 不存在 xyz axis 翻转；
- 不存在明显双层表面。

### Mask

- 背景没有大面积进入模型；
- 物体轮廓附近没有明显背景墙/桌面残留。

### Incremental

例如保存：

```text
model_kf_001.ply
model_kf_010.ply
model_kf_030.ply
model_kf_060.ply
```

人工确认模型随着 keyframe 增加而逐渐完整。

---

# 16. Phase 9 — 增加 ICP pose refinement

只有 baseline 成功后实现。

文件：

```text
registration/icp_refiner.py
```

接口：

```python
@dataclass
class ICPResult:
    success: bool
    T_world_cam_refined: np.ndarray

    fitness: float
    inlier_rmse: float

    translation_correction_m: float
    rotation_correction_deg: float


class ICPRefiner:

    def refine(
        self,
        current_object_pcd_cam,
        model_pcd_world,
        T_world_cam_initial,
    ) -> ICPResult:
        ...
```

含义：

```text
source = 当前 RealSense2 object PCD，仍在 C2 frame
target = 当前已经融合出的 object model，WORLD frame

initial transform =
T_world_cam2_robot
```

优先实现：

```text
point-to-plane ICP
```

ICP target 必须是：

```text
融合当前帧之前的模型
```

不能先 integrate 当前 frame 再拿自己配准自己。

---

# 17. ICP 安全约束

不能无条件接受 ICP。

增加：

```yaml
icp:
  enabled: false

  max_correspondence_distance_m: 0.01

  min_fitness: 0.4
  max_rmse_m: 0.008

  max_translation_correction_m: 0.02
  max_rotation_correction_deg: 5.0
```

接受条件：

```python
if (
    fitness >= min_fitness
    and rmse <= max_rmse
    and correction_translation <= threshold
    and correction_rotation <= threshold
):
    use refined pose
else:
    fallback robot pose
```

这是必须实现的 fallback。

机器人 pose 是强先验。

ICP 只是局部微调，不能让 ICP 自由把相机拉到很远的位置。

---

# 18. Phase 10 — 对比 Robot Pose vs ICP Pose

离线数据非常适合做 A/B test。

支持：

```bash
python tools/run_offline_reconstruction.py \
    --config configs/offline.yaml \
    --icp false
```

以及：

```bash
python tools/run_offline_reconstruction.py \
    --config configs/offline.yaml \
    --icp true
```

输出：

```text
output/run_robot_pose/
output/run_icp_pose/
```

记录：

```text
ICP fitness
ICP rmse
Δ translation
Δ rotation
```

重点观察：

```text
TSDF surface thickness
double surface
edge sharpness
cam1/cam2 overlap
```

---

# 19. Phase 11 — Pose Graph / Offline Global Optimization

这不是 MVP 的必需项。

当下面情况存在时再实现：

```text
机械臂绕物体一圈后
回到最初视角
但模型出现明显 loop drift
```

保存所有 keyframe：

```python
@dataclass
class Keyframe:
    index: int
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray

    T_world_cam_robot: np.ndarray
    T_world_cam_refined: np.ndarray
```

建立：

```text
nodes = camera2 keyframe poses

edges:
adjacent RGB-D ICP
overlap ICP
loop closure
robot pose prior
```

全局优化后：

```text
optimized poses
    ↓
创建新的 TSDFVolume
    ↓
从头 re-integrate 所有 keyframes
```

不要试图直接“移动已经融合过的 TSDF”。

---

# 20. Phase 12 — ReconstructionPipeline 抽象

此阶段将 offline / realtime 解耦。

定义：

```python
class FrameSource(Protocol):

    def start(self):
        ...

    def __iter__(self):
        yield FramePacket

    def stop(self):
        ...
```

两个实现：

```text
OfflineFrameSource
RealSenseFrameSource
```

核心：

```python
class ReconstructionPipeline:

    def process(self, packet: FramePacket):
        ...
```

因此：

## 离线

```python
source = OfflineFrameSource(...)
pipeline.run(source)
```

## 实时

```python
source = RealSenseFrameSource(...)
pipeline.run(source)
```

TSDF / SAM / ICP / keyframe / visualization 完全不改变。

---

# 21. Phase 13 — 接入 RealSense 实时采集

离线稳定后再创建：

```text
data/realsense_source.py
```

它只负责输出统一：

```python
FramePacket
```

不要把 pyrealsense2 API 泄漏到 reconstruction 模块。

实时 RealSense 运行于 Docker 时还需要 USB 设备透传、设备权限和 pyrealsense2/librealsense
兼容性验证。Task 12 应先决定使用宿主机采集进程向容器传输 `FramePacket`，还是向容器
显式映射 `/dev/bus/usb`；不要为了省事默认使用 `--privileged`。这不影响当前只挂载
离线 `data/` 的 MVP 镜像。

实时版本：

```text
RealSense1
RealSense2
Robot pose
     ↓
sync/alignment
     ↓
FramePacket
     ↓
ReconstructionPipeline
```

第一版实时仍然：

```text
单线程
同步 pipeline
```

确保结果正确。

---

# 22. Phase 14 — 实时性能优化

确认实时结果正确以后再做并行。

推荐 pipeline：

```text
Capture Thread
      ↓ Queue
SAM 2.1 StreamTracker Worker
      ↓ Queue
Geometry / Keyframe Worker
      ↓
ICP
      ↓
TSDF
      ↓
Visualization
```

原则：

```text
宁可丢 frame
不要让实时系统无限积压 frame
```

Queue：

```text
maxsize = 1~3
```

如果处理不过来：

```text
drop old frames
keep newest frame
```

因为 reconstruction 本来就只需要 keyframes。

---

# 23. 配置文件设计

`configs/offline.yaml`

```yaml
dataset:
  root: "/workspace/data"
  layout: "current_flat"
  cam1_prefix: "head"
  cam2_prefix: "wrist"
  frame_count_expected: 113
  pose_source: "JointStates.txt"
  # 已确认（null 时数据检查 fail fast）。
  pose_semantics: "T_base_tcp"
  quaternion_order: "xyzw"
  handeye_convention: "wrist_cam2=T_tcp_cam2,base_cam1=T_base_cam1"

world_frame: "head_optical"

devices:
  sam2: "cuda"
  tsdf: "CPU:0"

depth:
  # 必须由采集配置确认；16-bit PNG 本身不能证明单位。
  scale: null
  min_m: 0.15
  max_m: 1.2

mask:
  source: "cache"
  erosion_px: 3
  min_area_px: 1000

sam2:
  enabled: true
  checkpoint: "/models/sam2.1_hiera_tiny.pt"
  model_type: "tiny"
  mode: "video"
  object_id: 1
  prompt_type: "box"
  staging_root: "./output/sam2_frames"
  run_streams_sequentially: true
  offload_video_to_cpu: true
  offload_state_to_cpu: false
  cache_masks: true

cam1:
  initial_frames: 10
  update_interval_frames: 30
  max_integrations: 30

keyframe:
  translation_m: 0.02
  rotation_deg: 5.0
  min_valid_depth_ratio: 0.7

tsdf:
  device: "CPU:0"
  voxel_size_m: 0.002
  block_resolution: 16
  block_count: 50000
  trunc_voxel_multiplier: 4.0
  depth_max_m: 1.2
  weight_threshold: 3.0

icp:
  enabled: false
  max_correspondence_distance_m: 0.01
  min_fitness: 0.4
  max_rmse_m: 0.008
  max_translation_correction_m: 0.02
  max_rotation_correction_deg: 5.0

visualization:
  enabled: true
  update_every_keyframes: 5

output:
  root: "./output"
  save_keyframes: true
  save_debug: true
```

---

# 24. CLI

以下是 Task 实现完成后的目标 CLI；对应 `tools/` 和 `configs/` 当前尚不存在，因此现在
不可直接运行。每个命令应随对应 Task 落地后再加入 smoke test。

## 检查数据

```bash
python tools/inspect_dataset.py \
    --config configs/offline.yaml
```

## 预生成 mask

```bash
python tools/precompute_masks.py \
    --config configs/offline.yaml
```

## 验证坐标系

```bash
python tools/validate_transforms.py \
    --config configs/offline.yaml
```

## baseline

```bash
python tools/run_offline_reconstruction.py \
    --config configs/offline.yaml
```

## 开 ICP

```bash
python tools/run_offline_reconstruction.py \
    --config configs/offline.yaml \
    --icp
```

## 导出

```bash
python tools/export_model.py \
    --run output/run_xxx \
    --pointcloud \
    --mesh
```

---

# 25. 必须写的单元测试

## test_transforms.py

测试：

```text
identity
inverse
composition
camera-to-world
world-to-camera
```

必须满足：

```python
T @ inv(T) ≈ I
```

---

## test_projection.py

创建已知 3D 点：

```text
P = [0, 0, 1]
```

验证：

```text
project
unproject
```

结果一致。

---

## test_keyframe_selector.py

测试：

```text
0 mm / 0 deg
10 mm
30 mm
3 deg
10 deg
```

是否正确触发。

---

# 26. Debug 功能必须优先于算法复杂度

每个 keyframe 保存：

```text
rgb1
depth1 visualization
mask1

rgb2
depth2 visualization
mask2

cam2 raw object pcd
cam2 world object pcd

T_world_cam2_robot
T_world_cam2_refined
```

推荐目录：

```text
output/debug/keyframes/000012/
├── rgb2.png
├── depth2.png
├── mask2.png
├── raw_cam2.ply
├── world_cam2.ply
├── pose_robot.txt
└── pose_refined.txt
```

遇到模型错误时可以直接定位某一个 keyframe。

---

# 27. Cursor 实现顺序

Cursor 不要一次性生成完整系统。

严格按照以下顺序。

## Task 0

先建立可复现基线：

```text
将 src/segmention/sam2 纳入 Git 或固定为 submodule commit
创建根 Dockerfile、compose.yaml、requirements.txt 和最小 Python 包
构建 SAM2/cu121 镜像并验证 RTX 3060 可见
在容器 smoke test 后锁定 reconstruction 依赖版本
确认 checkpoint 通过只读 volume 提供
```

Task 0 不实现重建算法。

---

## Task 1

实现：

```text
models.py
offline_source.py
inspect_dataset.py
```

验收后再继续。

---

## Task 2

实现：

```text
transforms.py
pointcloud.py
validate_transforms.py
```

目标：

```text
cam1 + moving cam2 point cloud
正确统一到 world frame
```

这是第一个关键 milestone。

---

## Task 3

实现：

```text
sam2_segmenter.py
mask_cache.py
precompute_masks.py
```

目标：

```text
完整序列生成 object A masks
```

---

## Task 4

实现：

```text
rgbd.py
```

完成：

```text
mask erosion
depth filtering
masked RGB-D
```

---

## Task 5

实现：

```text
tsdf_volume.py
```

先用：

```text
cam1 单视角
```

验证 TSDF point cloud。

---

## Task 6

加入：

```text
cam2 known poses
```

不做 keyframe、不做 ICP，低频采样融合。

验证多视角重建。

---

## Task 7

实现：

```text
keyframe_selector.py
```

验证 incremental reconstruction。

这是第二个关键 milestone。

---

## Task 8

实现：

```text
live_viewer.py
save point cloud / mesh
logging
```

完成离线 MVP。

**状态（2026-08-20）**：代码已实现，待在目标容器（X11）实测后勾选验收项。
`visualization/live_viewer.py`（点云 + cam1/cam2 坐标系 + cam2 轨迹，热键
SPACE/S/M/Q，每 `visualization.update_every_keyframes` 个关键帧刷新一次）；
`TSDFVolume.save_mesh` / `save_point_cloud` / `block_count`；
`utils/logging.py`（控制台 + output/logs/run_*.log）；pipeline 增量快照
`model_kf_XXX.ply` 与逐关键帧 debug 记录（output/debug/fusion_debug.jsonl，
含 §15 要求的全部字段）。无显示环境用 `--no-viewer`。

---

## Task 9

实现：

```text
icp_refiner.py
```

增加：

```text
robot pose initialization
ICP refinement
safety fallback
```

---

## Task 10

做：

```text
ICP ON/OFF A/B comparison
```

确认 ICP 确实改善模型后才默认开启。

---

## Task 11

抽象：

```text
FrameSource
ReconstructionPipeline
```

确认 OfflineFrameSource 没有功能回归。

---

## Task 12

实现：

```text
RealSenseFrameSource
```

接实时数据。

---

## Task 13

性能优化：

```text
thread
queue
GPU
frame dropping
```

---

# 28. MVP 完成定义

只有同时满足下面条件，才认为离线 MVP 跑通。

- [x] 可以完整读取离线同步数据。
- [x] `pose_semantics`、`quaternion_order`、`handeye_convention` 和 `depth.scale` 均已确认，配置不再 fail fast。
- [x] `base_cam1` 标定修复并通过 validate_transforms 跨相机验收（2026-08-19 重标定后跨相机重叠 0.34）。
- [x] camera1 / camera2 内参正确。
- [x] `T_world_cam2` convention 完全明确。
- [x] Cam2 运动时，物体 world PCD 基本保持稳定。
- [x] SAM 2.1 可以在 RTX 3060 容器中稳定获得物体 A mask（2026-08-20 实测，双流 113 帧 mask 人工确认正确）。
- [x] SAM 2.1 真实 checkpoint smoke test 在目标容器中通过（sam2.1_hiera_tiny.pt）。
- [x] mask 可缓存（output/masks/cam{1,2}/，precompute_masks 一次性生成）。
- [x] masked RGB-D 正确（validate_tsdf_cam1 单视角表面人工确认为物体 A）。
- [x] Open3D TSDF 可以逐 keyframe integrate。
- [x] Open3D `CPU:0` VoxelBlockGrid smoke test 在目标容器中通过。
- [x] cam1 + cam2 可以融合（2026-08-20 实测：13 次 cam1 积分 + 9 个 cam2 关键帧，6862 点；跨视角残余错位待 Task 9 ICP 收敛）。
- [x] Cam2 移动时模型逐渐完整（比 cam1 单视角多出一块，颜色轮廓与 mask 对应）。
- [ ] 可以实时/准实时显示当前 point cloud。
- [x] 可以保存 `.ply` point cloud（output/pointcloud/object_a.ply + 增量快照 model_kf_XXX.ply）。
- [ ] 可以保存 `.ply` mesh。
- [x] 所有参数均从 YAML 配置读取。
- [ ] ICP 可以通过 config 完全关闭。
- [ ] 不使用 ICP 时 pipeline 仍可正常工作。

---

# 29. 第一阶段明确不做的内容

为了避免 Cursor 把工程做复杂，MVP 阶段不要实现：

```text
NeRF
BundleSDF
Neural SDF
Next-Best-View
机械臂轨迹规划
ROS
复杂 GUI
数据库
Web server
全局 feature matching
FPFH
无初值 global registration
```

这些都属于后续扩展。

---

# 30. 下一阶段扩展路线

离线 TSDF + ICP 稳定以后：

```text
Stage A
Known pose TSDF
        ↓
Stage B
Robot pose + ICP
        ↓
Stage C
Pose graph + reintegration
        ↓
Stage D
Neural SDF / BundleSDF
        ↓
Stage E
TSDF uncertainty / coverage map
        ↓
Stage F
Next-Best-View
        ↓
Stage G
机械臂主动移动 RealSense2
```

最终形成：

```text
固定 RealSense1
        ↓
持续观察物体
        │
        ├────────────┐
        ↓            │
Current Model        │
        ↓            │
Coverage /           │
Uncertainty          │
        ↓            │
Next-Best-View       │
        ↓            │
Robot Motion         │
        ↓            │
RealSense2 ──────────┘
```

---

# 31. 给 Cursor Agent 的总体开发约束

在 Cursor 中执行本计划时遵守：

1. 一次只实现当前 Task，不提前实现后续阶段。
2. 所有坐标变换必须使用 `T_A_B` 命名规范。
3. reconstruction 内部统一使用 `T_world_cam`。
4. Open3D extrinsic convention 只允许在 Open3D wrapper 内转换。
5. 禁止在不同模块随意 `np.linalg.inv()`。
6. 禁止硬编码 intrinsics、depth scale、voxel size。
7. 所有关键参数进入 YAML。
8. 每增加一个复杂算法必须保留关闭开关。
9. ICP 失败必须回退到 robot pose。
10. 优先保证可 debug，再优化速度。
11. Offline 和 Realtime 必须共享 `FramePacket` 和 `ReconstructionPipeline`。
12. 第一版同步单线程实现，正确后再并行优化。
13. 每完成一个 Task 都运行对应测试和最小 demo。
14. 坐标系 baseline 没通过之前禁止开始 TSDF。
15. TSDF baseline 没通过之前禁止开始 ICP。
