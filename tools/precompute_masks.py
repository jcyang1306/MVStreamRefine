"""Precompute object-A masks for the sole wrist-camera stream with SAM 2.1.

JPEGs are staged into a numeric symlink directory, tracking is anchored by a
manual first-frame xyxy box prompt, and every resulting mask is written to the
disk cache (output/masks/cam/*.png + mask_metadata.jsonl). Reconstruction
later reads this cache instead of re-running SAM 2.1.

Video tracking has no quality scores, so per-frame metadata records mask area,
IoU with the previous frame and valid-depth ratio; a few RGB overlay previews
are saved for human spot checks.

Requires torch + sam2-inference + checkpoint (run inside the CUDA container).
If the camera box is omitted, its first RGB frame is shown for interactive ROI
selection (drag the rectangle, then press Enter/Space; press C to cancel):
    python tools/precompute_masks.py --config configs/offline.yaml \
        [--cam-box X1 Y1 X2 Y2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.segmentation.mask_cache import MaskCache
from object_reconstruction.segmentation.sam2_segmenter import (
    SAM2Segmenter,
    Sam2Prompt,
    release_gpu_memory,
    stage_jpeg_sequence,
)
from object_reconstruction.utils.config import load_config

def select_interactive_box(
    image_path: str | Path, camera_name: str
) -> tuple[float, float, float, float]:
    """Display an image and let the user select an xyxy rectangle with OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "interactive box selection requires opencv-python with GUI support; "
            "rebuild the project image after updating requirements.txt"
        ) from exc

    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read first-frame RGB image: {image_path}")

    window_name = f"Select object A - {camera_name}"
    print(
        f"[{camera_name}] select object A in {image_path}: "
        "drag ROI, press Enter/Space to confirm, or C to cancel"
    )
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        x, y, width, height = cv2.selectROI(
            window_name, image, showCrosshair=True, fromCenter=False
        )
    except cv2.error as exc:
        raise RuntimeError(
            "could not open the ROI selection window. When running in Docker, "
            "pass DISPLAY and mount /tmp/.X11-unix (see README.md)."
        ) from exc
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Headless OpenCV may reject both window creation and cleanup;
            # preserve the actionable error raised above.
            pass

    if width <= 0 or height <= 0:
        raise RuntimeError(f"{camera_name} ROI selection was cancelled or empty")

    box = (float(x), float(y), float(x + width), float(y + height))
    print(f"[{camera_name}] selected xyxy box: {' '.join(f'{v:g}' for v in box)}")
    return box


def resolve_box(
    source: OfflineFrameSource,
    cam_box: list[float] | None,
) -> tuple[float, float, float, float]:
    """Use the CLI box when present; otherwise select it interactively."""
    if cam_box is not None:
        return tuple(cam_box)
    first_index = source.indices[0]
    prefix = source.cam_prefix
    image_path = source.root / f"frame-{first_index:06d}_{prefix}_color.jpg"
    return select_interactive_box(image_path, f"cam ({prefix})")


def load_depth(source: OfflineFrameSource, prefix: str, index: int) -> np.ndarray:
    return np.asarray(Image.open(source.root / f"frame-{index:06d}_{prefix}_depth.png"))


def iou(a: np.ndarray | None, b: np.ndarray) -> float | None:
    if a is None:
        return None
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def save_overlay(
    source: OfflineFrameSource, prefix: str, index: int, mask: np.ndarray, path: Path
) -> None:
    rgb = np.asarray(
        Image.open(source.root / f"frame-{index:06d}_{prefix}_color.jpg")
    ).astype(np.float64)
    rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array([255.0, 0.0, 0.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def run_stream(
    box: tuple[float, float, float, float],
    config: dict,
    source: OfflineFrameSource,
    cache: MaskCache,
    preview_every: int,
) -> list[dict]:
    prefix = source.cam_prefix
    staging_dir = Path(config["sam2"]["staging_root"]) / prefix
    stage_jpeg_sequence(source.root, prefix, source.indices, staging_dir)
    print(f"[cam] staged {len(source.indices)} frames -> {staging_dir}")

    depth_cfg = config["depth"]
    depth_scale = float(depth_cfg["scale"])
    preview_dir = cache.root / "previews"

    segmenter = SAM2Segmenter.from_config(config)
    prompt = Sam2Prompt(object_id=config["sam2"].get("object_id", 1), box=box)

    records: list[dict] = []
    previous_mask: np.ndarray | None = None
    for frame_index, mask in segmenter.track_sequence(staging_dir, prompt):
        cache.save(frame_index, mask)

        area = int(mask.sum())
        depth_m = load_depth(source, prefix, frame_index).astype(np.float64) / depth_scale
        in_range = (depth_m >= depth_cfg["min_m"]) & (depth_m <= depth_cfg["max_m"])
        valid_depth_ratio = float(in_range[mask].mean()) if area else 0.0
        records.append(
            {
                "frame": int(frame_index),
                "mask_area": area,
                "valid_depth_ratio": round(valid_depth_ratio, 4),
                "iou_prev": None if (v := iou(previous_mask, mask)) is None else round(v, 4),
            }
        )
        previous_mask = mask

        if frame_index % preview_every == 0 or frame_index == source.indices[-1]:
            save_overlay(
                source, prefix, frame_index, mask,
                preview_dir / f"cam_{frame_index:06d}.png",
            )

    del segmenter
    release_gpu_memory()
    return records


def summarize(records: list[dict], min_area_px: int, total_frames: int) -> bool:
    areas = [r["mask_area"] for r in records]
    small = [r["frame"] for r in records if r["mask_area"] < min_area_px]
    ious = [r["iou_prev"] for r in records if r["iou_prev"] is not None]
    print(
        f"[cam] masks {len(records)}/{total_frames}  "
        f"area min/mean {min(areas)}/{int(np.mean(areas))}  "
        f"iou_prev min/mean {min(ious):.3f}/{np.mean(ious):.3f}"
    )
    if small:
        print(f"[cam] WARNING: {len(small)} frames below min_area_px={min_area_px}: "
              f"{small[:10]}{'...' if len(small) > 10 else ''}")
    ok = len(records) == total_frames and min(areas) > 0
    if not ok:
        print("[cam] FAIL: missing or empty masks")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cam-box", type=float, nargs=4,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="first-frame xyxy box; omitted = interactive selection")
    parser.add_argument("--preview-every", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    if not config["sam2"].get("enabled", True):
        print("FAIL: sam2.enabled is false in config")
        return 1
    source = OfflineFrameSource.from_config(config)
    cache = MaskCache(Path(config["output"]["root"]) / "masks")
    box = resolve_box(source, args.cam_box)
    records = run_stream(box, config, source, cache, args.preview_every)
    ok = summarize(records, config["mask"]["min_area_px"], len(source))

    cache.write_metadata(records)
    print(f"metadata: {cache.metadata_path}")
    print(f"previews: {cache.root / 'previews'} (human spot check)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
