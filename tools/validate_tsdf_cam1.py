"""Task 5 acceptance: single-view TSDF fusion from cam1 (head, fixed = WORLD).

Integrates the first N cam1 frames (masked via the precomputed SAM cache) into
a TSDFVolume with T_world_cam = I, then exports the extracted point cloud to
output/debug/tsdf_cam1.ply and prints a summary (point count, bounding box).
Exits non-zero when the point cloud is empty or a cached mask is missing.

Requires open3d, so run inside the CUDA container:
    docker compose run --rm reconstruction \
        python3 tools/validate_tsdf_cam1.py --config configs/offline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.fusion.tsdf_volume import TSDFVolume
from object_reconstruction.geometry.rgbd import preprocess_object_rgbd
from object_reconstruction.segmentation.mask_cache import MaskCache
from object_reconstruction.utils.config import load_config


def point_positions(point_cloud) -> np.ndarray:
    """Positions of an open3d.t PointCloud as (N, 3); empty clouds give (0, 3)."""
    try:
        return point_cloud.point.positions.numpy()
    except KeyError:
        return np.zeros((0, 3), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", type=int, default=10,
                        help="number of leading frames to integrate "
                             "(cam1 is fixed, so a few frames suffice)")
    parser.add_argument("--output", default=None,
                        help="output .ply path; default <output.root>/debug/tsdf_cam1.ply")
    args = parser.parse_args()
    if args.frames <= 0:
        print(f"FAIL: --frames must be > 0, got {args.frames}")
        return 1

    config = load_config(args.config)
    source = OfflineFrameSource.from_config(config)
    cache = MaskCache(Path(config["output"]["root"]) / "masks")
    tsdf = TSDFVolume.from_config(config)
    print(f"tsdf: device={tsdf.params.device} voxel_size_m={tsdf.params.voxel_size_m} "
          f"depth_max_m={tsdf.params.depth_max_m} "
          f"weight_threshold={tsdf.params.weight_threshold}")

    for position in range(min(args.frames, len(source))):
        packet = source.read_packet(position)
        mask = cache.get_cam1(packet.index)
        if mask is None:
            print(f"FAIL: no cached cam1 mask for frame {packet.index} "
                  f"(expected {cache.mask_path(1, packet.index)}); "
                  "run tools/precompute_masks.py first")
            return 1
        rgbd = preprocess_object_rgbd(packet.cam1.rgb, packet.cam1.depth, mask, config)
        if rgbd.valid_depth_px == 0:
            print(f"[frame {packet.index}] WARNING: no valid object depth, skipped")
            continue
        # cam1 defines WORLD, so its pose is the identity for every frame.
        tsdf.integrate(rgbd.rgb, rgbd.depth, packet.cam1.intrinsics,
                       T_world_cam=np.eye(4))
        print(f"[frame {packet.index}] integrated  mask_area={rgbd.mask_area_px}  "
              f"valid_depth_ratio={rgbd.valid_depth_ratio:.3f}")

    if tsdf.integration_count == 0:
        print("FAIL: no frame had valid object depth; nothing was integrated")
        return 1

    points = point_positions(tsdf.extract_point_cloud())
    if len(points) == 0:
        print("FAIL: extracted point cloud is empty "
              "(check masks, depth range and weight_threshold)")
        return 1

    output_path = (Path(args.output) if args.output
                   else Path(config["output"]["root"]) / "debug" / "tsdf_cam1.ply")
    tsdf.save(output_path)

    mins, maxs = points.min(axis=0), points.max(axis=0)
    extent = maxs - mins
    print(f"integrated frames: {tsdf.integration_count}")
    print(f"points: {len(points)}")
    print(f"bbox min  [m]: {mins[0]:+.4f} {mins[1]:+.4f} {mins[2]:+.4f}")
    print(f"bbox max  [m]: {maxs[0]:+.4f} {maxs[1]:+.4f} {maxs[2]:+.4f}")
    print(f"bbox size [m]: {extent[0]:.4f} {extent[1]:.4f} {extent[2]:.4f}")
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
