"""Single wrist-camera TSDF smoke test in robot-base WORLD.

Selects and integrates the first N valid camera keyframes with their real
robot-derived poses T_world_cam, then exports output/debug/tsdf_cam.ply.
Exits non-zero when masks are missing, no keyframe is integrated, or the
extracted point cloud is empty.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.fusion.keyframe_selector import KeyframeSelector
from object_reconstruction.fusion.tsdf_volume import TSDFVolume
from object_reconstruction.geometry.rgbd import preprocess_object_rgbd
from object_reconstruction.segmentation.mask_cache import MaskCache
from object_reconstruction.utils.config import load_config


def point_positions(point_cloud) -> np.ndarray:
    try:
        return point_cloud.point.positions.numpy()
    except KeyError:
        return np.zeros((0, 3), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--keyframes", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.keyframes <= 0:
        print(f"FAIL: --keyframes must be > 0, got {args.keyframes}")
        return 1

    config = load_config(args.config)
    source = OfflineFrameSource.from_config(config)
    cache = MaskCache(Path(config["output"]["root"]) / "masks")
    selector = KeyframeSelector.from_config(config)
    tsdf = TSDFVolume.from_config(config)
    print(
        f"tsdf: device={tsdf.params.device} "
        f"voxel_size_m={tsdf.params.voxel_size_m} "
        f"weight_threshold={tsdf.params.weight_threshold}"
    )

    for packet in source:
        mask = cache.get(packet.index)
        if mask is None:
            print(
                f"FAIL: no cached cam mask for frame {packet.index} "
                f"(checked {cache.mask_path(packet.index)} and legacy "
                f"{cache.legacy_mask_path(packet.index)}); "
                "run tools/precompute_masks.py first"
            )
            return 1
        if packet.T_world_cam is None:
            print("FAIL: T_world_cam is unavailable; confirm pose/handeye conventions")
            return 1

        rgbd = preprocess_object_rgbd(
            packet.cam.rgb, packet.cam.depth, mask, config
        )
        if not selector.should_add(
            packet.T_world_cam, rgbd.mask_area_px, rgbd.valid_depth_ratio
        ):
            continue
        tsdf.integrate(
            rgbd.rgb,
            rgbd.depth,
            packet.cam.intrinsics,
            T_world_cam=packet.T_world_cam,
        )
        print(
            f"[frame {packet.index}] keyframe #{tsdf.integration_count} integrated  "
            f"mask_area={rgbd.mask_area_px}  "
            f"valid_depth_ratio={rgbd.valid_depth_ratio:.3f}"
        )
        if tsdf.integration_count >= args.keyframes:
            break

    if tsdf.integration_count == 0:
        print("FAIL: no keyframe had valid object depth; nothing was integrated")
        return 1

    points = point_positions(tsdf.extract_point_cloud())
    if len(points) == 0:
        print(
            "FAIL: extracted point cloud is empty "
            "(check masks, depth range and weight_threshold)"
        )
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else Path(config["output"]["root"]) / "debug" / "tsdf_cam.ply"
    )
    tsdf.save_point_cloud(output_path)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    extent = maxs - mins
    print(f"integrated keyframes: {tsdf.integration_count}")
    print(f"points: {len(points)}")
    print(f"bbox min  [m]: {mins[0]:+.4f} {mins[1]:+.4f} {mins[2]:+.4f}")
    print(f"bbox max  [m]: {maxs[0]:+.4f} {maxs[1]:+.4f} {maxs[2]:+.4f}")
    print(f"bbox size [m]: {extent[0]:.4f} {extent[1]:.4f} {extent[2]:.4f}")
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
