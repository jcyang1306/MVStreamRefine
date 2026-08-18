# SAM2 最小推理包

这是一个可从 SAM2 主仓库单独复制、构建和安装的 SAM 2.1 推理子项目。它只包含图片提示分割、离线视频传播和逐帧流式跟踪所需代码，不包含训练、Demo、AMG 或 benchmark。

## 安装

```bash
pip install .
# MP4 输入另装解码依赖；JPEG 帧目录不需要此 extra
pip install ".[video]"
```

构造 API 时必须传入本地 checkpoint。`model_type` 可取 `tiny`、`small`、`base_plus`、`large`，分别映射到随 wheel 发布的 SAM 2.1 配置。`device="auto"` 依次选择 CUDA、MPS、CPU。

本包为自包含发行物，内部自带精简后的 `sam2` 模块。请在独立虚拟环境或 Docker
镜像中安装，不要与官方 `SAM-2` wheel 同时安装，否则两个发行物会占用同一个
Python 模块名。

## 图片

```python
import numpy as np
from PIL import Image
from sam2_inference import ImageSegmenter

rgb = np.asarray(Image.open("image.jpg").convert("RGB"))
segmenter = ImageSegmenter("sam2.1_hiera_tiny.pt", model_type="tiny")
result = segmenter.segment(rgb, box=[100, 80, 420, 360], multimask=False)
mask = result.masks[0]       # bool, H x W
score = result.scores[0]     # 预测质量
logits = result.logits[0]    # 低分辨率 logits，可用于迭代提示
```

`segment` 接受像素坐标的 `points`（N×2）与对应的 0/1 `labels`，也接受 `box`（xyxy）；点和框可以组合。

## 离线视频

```python
from sam2_inference import VideoTracker

tracker = VideoTracker("sam2.1_hiera_tiny.pt", model_type="tiny")
tracker.open("frames")  # 编号 JPEG 目录，或安装 video extra 后传 MP4
tracker.add_prompt(0, object_id=1, box=[100, 80, 420, 360])
for result in tracker.track():
    print(result.frame_index, result.object_ids, result.masks.shape)
tracker.close()
```

`add_prompt` 支持点/标签、框或二维 bool 掩码。`TrackingResult.logits` 与 `masks` 均为源视频分辨率，首维与 `object_ids` 对齐；视频预测器不提供质量分数，因此 `scores` 为 `None`。

## RGB 流

```python
from sam2_inference import StreamTracker

tracker = StreamTracker("sam2.1_hiera_tiny.pt", model_type="tiny")
initial = tracker.add_prompt(first_rgb, object_id=1, box=[100, 80, 420, 360])
for rgb in incoming_rgb_frames:
    result = tracker.track(rgb)
    consume(result.masks[0])
tracker.close()
```

流式 API 只接收 `uint8`、H×W×3 的 RGB NumPy 数组，不依赖 OpenCV 或 RealSense。一个会话内分辨率必须保持不变。可在首帧多次调用 `add_prompt` 使用不同 `object_id` 初始化多对象；单对象路径为稳定的最低保证。旧的非条件帧会被裁剪以限制内存。

## 可选 CUDA 扩展

普通 wheel 默认不编译 connected-components CUDA 扩展，因此没有 nvcc 也能构建。扩展只影响小孔填充后处理；缺失时官方内核会跳过该步骤。

```bash
# 先在当前环境安装与 CUDA 匹配的 torch
SAM2_BUILD_CUDA=1 pip install --no-build-isolation .
```

## Docker

Docker 镜像基于 CUDA 12.1 runtime，并固定 torch 2.5.1/cu121，兼容 NVIDIA 535 驱动：

```bash
docker build -t sam2-inference .
docker run --rm --gpus all sam2-inference
```

## 测试

```bash
pip install ".[test]"
pytest
python -m build
```

无权重测试会 mock 官方 predictor。设置 `SAM2_CHECKPOINT=/path/to/checkpoint.pt` 后会额外运行真实 tiny checkpoint 的图片、JPEG 目录和两帧流式 smoke tests。
