"""Validate the single wrist-camera transform chain in robot-base WORLD.

For sampled frames, back-projects wrist depth and transforms it with

    T_world_cam = T_base_tcp @ T_tcp_cam

Static-scene points should remain registered while the robot moves. The same
metric is computed with an inverted hand-eye matrix; the confirmed convention
must win. No fixed camera or cross-camera metric is used.

Outputs:
    output/debug/transform_validation.json
    output/debug/transforms/cam_world_*.ply
    output/debug/transforms/cam_trajectory.ply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.calibration.transforms import (
    compose_T_world_cam,
    invert_transform,
    pose7d_to_matrix,
)
from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.geometry.pointcloud import depth_to_pointcloud, transform_points
from object_reconstruction.utils.config import load_config
from object_reconstruction.utils.io import write_ply_points

VOXEL_SIZE_M = 0.02
VARIANTS = {"confirmed": False, "flip_tcp_cam": True}


def voxel_set(points: np.ndarray) -> set[tuple[int, int, int]]:
    if len(points) == 0:
        return set()
    return set(map(tuple, np.floor(points / VOXEL_SIZE_M).astype(np.int64)))


def overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--pixel-stride", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    depth_cfg = config["depth"]
    if depth_cfg.get("scale") is None:
        print("FAIL: depth.scale is null; confirm conventions before validation")
        return 1

    source = OfflineFrameSource.from_config(config)
    if not source.conventions_confirmed:
        print("FAIL: pose/handeye conventions are null; transform validation is blocked")
        return 1

    out_dir = Path(config["output"]["root"]) / "debug" / "transforms"
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = list(range(0, len(source), args.frame_stride))
    if positions[-1] != len(source) - 1:
        positions.append(len(source) - 1)

    T_base_tcp_all = {
        p: pose7d_to_matrix(source.raw_poses[p], source.quaternion_order)
        for p in positions
    }
    results: dict[str, dict[str, float]] = {}
    trajectory: list[np.ndarray] = []

    for variant, flip_tcp in VARIANTS.items():
        T_tcp_cam = (
            invert_transform(source.T_tcp_cam) if flip_tcp else source.T_tcp_cam
        )
        self_overlaps: list[float] = []
        reference_voxels = None
        for p in positions:
            packet = source.read_packet(p)
            T_world_cam = compose_T_world_cam(T_base_tcp_all[p], T_tcp_cam)
            points, colors = depth_to_pointcloud(
                packet.cam.depth,
                packet.cam.intrinsics,
                depth_scale=depth_cfg["scale"],
                depth_min_m=depth_cfg["min_m"],
                depth_max_m=depth_cfg["max_m"],
                stride=args.pixel_stride,
                rgb=packet.cam.rgb if variant == "confirmed" else None,
            )
            world_points = transform_points(points, T_world_cam)
            voxels = voxel_set(world_points)
            if reference_voxels is None:
                reference_voxels = voxels
            else:
                self_overlaps.append(overlap(voxels, reference_voxels))

            if variant == "confirmed":
                trajectory.append(T_world_cam[:3, 3])
                if p in (positions[0], positions[-1]):
                    index = source.indices[p]
                    write_ply_points(
                        out_dir / f"cam_world_{index:06d}.ply",
                        world_points,
                        colors,
                    )

        results[variant] = {
            "static_overlap_mean": float(np.mean(self_overlaps)),
            "static_overlap_min": float(np.min(self_overlaps)),
        }

    trajectory_array = np.asarray(trajectory)
    write_ply_points(
        out_dir / "cam_trajectory.ply",
        trajectory_array,
        np.tile([255, 0, 0], (len(trajectory_array), 1)),
    )

    confirmed = results["confirmed"]
    best_variant = max(results, key=lambda k: results[k]["static_overlap_mean"])
    passed = (
        best_variant == "confirmed"
        and confirmed["static_overlap_mean"] >= 0.5
    )
    report = {
        "world_frame": "robot_base",
        "voxel_size_m": VOXEL_SIZE_M,
        "frames_evaluated": [source.indices[p] for p in positions],
        "results_per_variant": results,
        "best_variant": best_variant,
        "passed": passed,
    }
    report_path = out_dir.parent / "transform_validation.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"frames evaluated: {len(positions)} (stride {args.frame_stride})")
    for variant, result in results.items():
        marker = " <= confirmed" if variant == "confirmed" else ""
        print(
            f"{variant:16s} static={result['static_overlap_mean']:.3f} "
            f"min={result['static_overlap_min']:.3f}{marker}"
        )
    print(
        f"cam trajectory span: {trajectory_array.min(axis=0)} "
        f".. {trajectory_array.max(axis=0)}"
    )
    print(f"report: {report_path}")
    print(
        "criterion (static-scene consistency in robot base): "
        f"{'PASS' if passed else 'FAIL'} "
        f"({confirmed['static_overlap_mean']:.3f}, best={best_variant})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
