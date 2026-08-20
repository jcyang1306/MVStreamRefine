"""Tasks 6-7 acceptance: dual-camera incremental reconstruction, no ICP.

Runs the OfflinePipeline over the full sequence: cam1 (WORLD anchor) at low
frequency plus cam2 keyframes at known poses T_world_cam2, all fused into one
TSDF volume. Exports the final object point cloud to
<output.root>/pointcloud/object_a.ply.

Requires open3d and the precomputed mask cache, so run inside the container:
    docker compose run --rm reconstruction \
        python3 tools/run_offline_reconstruction.py --config configs/offline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.fusion.keyframe_selector import KeyframeSelector
from object_reconstruction.fusion.tsdf_volume import TSDFVolume
from object_reconstruction.pipeline.offline_pipeline import OfflinePipeline
from object_reconstruction.segmentation.mask_cache import MaskCache
from object_reconstruction.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="process only the first N frames (debugging)")
    parser.add_argument("--output", default=None,
                        help="output .ply path; default "
                             "<output.root>/pointcloud/object_a.ply")
    args = parser.parse_args()

    config = load_config(args.config)
    source = OfflineFrameSource.from_config(config)
    cache = MaskCache(Path(config["output"]["root"]) / "masks")
    if not cache.frames(1) or not cache.frames(2):
        print(f"FAIL: mask cache incomplete under {cache.root} "
              "(need cam1/ and cam2/); run tools/precompute_masks.py first")
        return 1

    tsdf = TSDFVolume.from_config(config)
    selector = KeyframeSelector.from_config(config)
    pipeline = OfflinePipeline(tsdf, selector, config)
    print(f"tsdf: device={tsdf.params.device} voxel_size_m={tsdf.params.voxel_size_m}")
    print(f"keyframe: translation_m={selector.translation_m} "
          f"rotation_deg={selector.rotation_deg} "
          f"min_mask_area_px={selector.min_mask_area_px} "
          f"min_valid_depth_ratio={selector.min_valid_depth_ratio}")

    stats = pipeline.run(source, cache, max_frames=args.max_frames)

    print(f"frames processed:    {stats['frames_processed']}")
    print(f"cam1 integrations:   {stats['cam1_integrations']}")
    print(f"cam2 keyframes:      {stats['cam2_keyframes']} "
          f"(rejected {stats['cam2_rejected']})")
    if stats["missing_mask_frames"]:
        print(f"WARNING: {len(stats['missing_mask_frames'])} frames skipped for "
              f"missing masks: {stats['missing_mask_frames'][:10]}"
              f"{'...' if len(stats['missing_mask_frames']) > 10 else ''}")
    if stats["cam2_keyframes"] == 0:
        print("FAIL: no cam2 keyframe was integrated; multi-view reconstruction "
              "did not happen (check masks / keyframe thresholds)")
        return 1

    output_path = (Path(args.output) if args.output
                   else Path(config["output"]["root"]) / "pointcloud" / "object_a.ply")
    tsdf.save(output_path)

    points = tsdf.extract_point_cloud().point.positions.numpy()
    mins, maxs = points.min(axis=0), points.max(axis=0)
    print(f"points: {len(points)}")
    print(f"bbox size [m]: {maxs[0]-mins[0]:.4f} {maxs[1]-mins[1]:.4f} "
          f"{maxs[2]-mins[2]:.4f}")
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
