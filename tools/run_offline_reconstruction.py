"""Single wrist-camera incremental reconstruction in the robot-base frame.

Runs the OfflinePipeline over the full sequence: camera keyframes at poses
from robot kinematics, optionally refined per keyframe by point-to-plane ICP
with safety fallback.
A/B comparison (PLAN section 18): pass --icp true / --icp false to write
results into <output.root>/run_icp_pose/ or run_robot_pose/. Exports:
    <output.root>/pointcloud/object_a.ply        final point cloud
    <output.root>/mesh/object_a_mesh.ply         final triangle mesh
    <output.root>/pointcloud/model_kf_XXX.ply    incremental snapshots
    <output.root>/debug/fusion_debug.jsonl       per-keyframe debug records
    <output.root>/logs/run_*.log                 run log

With visualization.enabled (and an X11 display for Docker) a live viewer
shows the growing point cloud, robot-base frame and camera trajectory
(SPACE pause, S save point cloud, M save mesh, Q quit). Use --no-viewer to
force headless operation.

Requires open3d and the precomputed mask cache, so run inside the container:
    docker compose run --rm reconstruction \
        python3 tools/run_offline_reconstruction.py --config configs/offline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.fusion.keyframe_selector import KeyframeSelector
from object_reconstruction.fusion.tsdf_volume import TSDFVolume
from object_reconstruction.pipeline.offline_pipeline import OfflinePipeline
from object_reconstruction.segmentation.mask_cache import MaskCache
from object_reconstruction.utils.config import load_config
from object_reconstruction.utils.logging import setup_logging

logger = logging.getLogger("tools.run_offline_reconstruction")


def build_viewer(config: dict, tsdf: TSDFVolume, output_root: Path, no_viewer: bool):
    if no_viewer or not bool(config.get("visualization", {}).get("enabled", False)):
        return None
    from object_reconstruction.visualization.live_viewer import LiveViewer

    def save_point_cloud() -> None:
        path = output_root / "pointcloud" / f"model_{datetime.now():%H%M%S}.ply"
        tsdf.save_point_cloud(path)
        logger.info("hotkey save: %s", path)

    def save_mesh() -> None:
        path = output_root / "mesh" / f"model_{datetime.now():%H%M%S}.ply"
        tsdf.save_mesh(path)
        logger.info("hotkey save: %s", path)

    return LiveViewer(on_save_point_cloud=save_point_cloud, on_save_mesh=save_mesh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="process only the first N frames (debugging)")
    parser.add_argument("--output", default=None,
                        help="output point-cloud .ply path; default "
                             "<output.root>/pointcloud/object_a.ply")
    parser.add_argument("--no-viewer", action="store_true",
                        help="disable the live viewer (headless run)")
    parser.add_argument("--icp", choices=["true", "false"], default=None,
                        help="override icp.enabled; when given, results go to "
                             "<output.root>/run_icp_pose/ or run_robot_pose/ "
                             "for A/B comparison (PLAN section 18)")
    args = parser.parse_args()

    config = load_config(args.config)
    base_root = Path(config["output"]["root"])
    icp_enabled = bool(config.get("icp", {}).get("enabled", False))
    if args.icp is not None:
        icp_enabled = args.icp == "true"
        # A/B mode: results are separated per pose source; the mask cache and
        # SAM staging stay under the base root and are shared by both runs.
        output_root = base_root / ("run_icp_pose" if icp_enabled else "run_robot_pose")
        config["output"]["root"] = str(output_root)
    else:
        output_root = base_root
    log_path = setup_logging(output_root)
    logger.info("log file: %s", log_path)

    source = OfflineFrameSource.from_config(config)
    cache = MaskCache(base_root / "masks")
    if not cache.frames():
        logger.error("mask cache is empty under %s (need cam/ or legacy cam2/); "
                     "run tools/precompute_masks.py first", cache.root)
        return 1

    tsdf = TSDFVolume.from_config(config)
    selector = KeyframeSelector.from_config(config)
    viewer = build_viewer(config, tsdf, output_root, args.no_viewer)
    refiner = None
    if icp_enabled:
        from object_reconstruction.registration.icp_refiner import ICPRefiner
        refiner = ICPRefiner.from_config(config)
    pipeline = OfflinePipeline(tsdf, selector, config, viewer=viewer,
                               icp_refiner=refiner)
    logger.info("tsdf: device=%s voxel_size_m=%s",
                tsdf.params.device, tsdf.params.voxel_size_m)
    logger.info("keyframe: translation_m=%s rotation_deg=%s min_mask_area_px=%s "
                "min_valid_depth_ratio=%s",
                selector.translation_m, selector.rotation_deg,
                selector.min_mask_area_px, selector.min_valid_depth_ratio)
    logger.info("viewer: %s", "on" if viewer is not None else "off")
    if refiner is not None:
        logger.info("icp: on  max_corr=%.3fm min_fitness=%.2f max_rmse=%.3fm "
                    "max_dt=%.3fm max_dr=%.1fdeg",
                    refiner.params.max_correspondence_distance_m,
                    refiner.params.min_fitness, refiner.params.max_rmse_m,
                    refiner.params.max_translation_correction_m,
                    refiner.params.max_rotation_correction_deg)
    else:
        logger.info("icp: off (robot pose only)")

    stats = pipeline.run(source, cache, max_frames=args.max_frames)

    logger.info("frames processed:  %d", stats["frames_processed"])
    logger.info("keyframes:        %d (rejected %d)",
                stats["keyframes"], stats["rejected"])
    if refiner is not None:
        logger.info("icp: accepted %d / fallback %d",
                    stats["icp_accepted"], stats["icp_fallback"])
        icp_runs = [r["icp"] for r in stats["debug_records"]
                    if r.get("icp") and "fitness" in r["icp"]]
        if icp_runs:
            logger.info(
                "icp mean: fitness=%.3f rmse=%.1fmm dt=%.1fmm dr=%.2fdeg",
                sum(r["fitness"] for r in icp_runs) / len(icp_runs),
                sum(r["inlier_rmse"] for r in icp_runs) / len(icp_runs) * 1000,
                sum(r["translation_correction_m"] for r in icp_runs)
                / len(icp_runs) * 1000,
                sum(r["rotation_correction_deg"] for r in icp_runs) / len(icp_runs),
            )
    if stats["missing_mask_frames"]:
        logger.warning("%d frames skipped for missing masks: %s%s",
                       len(stats["missing_mask_frames"]),
                       stats["missing_mask_frames"][:10],
                       "..." if len(stats["missing_mask_frames"]) > 10 else "")
    if stats["quit_requested"]:
        logger.info("stopped early by viewer quit; saving current state")
    if stats["keyframes"] == 0:
        logger.error("no camera keyframe was integrated "
                     "(check masks / keyframe thresholds)")
        return 1

    if bool(config["output"].get("save_debug", False)) and stats["debug_records"]:
        debug_path = output_root / "debug" / "fusion_debug.jsonl"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8") as fh:
            for record in stats["debug_records"]:
                fh.write(json.dumps(record) + "\n")
        logger.info("debug records: %s (%d keyframes)",
                    debug_path, len(stats["debug_records"]))

    pc_path = (Path(args.output) if args.output
               else output_root / "pointcloud" / "object_a.ply")
    tsdf.save_point_cloud(pc_path)
    mesh_path = output_root / "mesh" / "object_a_mesh.ply"
    tsdf.save_mesh(mesh_path)

    points = tsdf.extract_point_cloud().point.positions.numpy()
    mins, maxs = points.min(axis=0), points.max(axis=0)
    logger.info("points: %d  tsdf_blocks: %d", len(points), tsdf.block_count())
    logger.info("bbox size [m]: %.4f %.4f %.4f",
                maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2])
    logger.info("saved point cloud: %s", pc_path)
    logger.info("saved mesh:        %s", mesh_path)
    if stats["snapshots"]:
        logger.info("snapshots: %s", ", ".join(stats["snapshots"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
