"""Phase 1 coordinate-transform validation (PLAN section 7). No TSDF, no SAM.

For sampled frames, back-projects cam1/cam2 depth into world-frame point
clouds using the confirmed conventions and checks the key acceptance
criterion: while cam2 (wrist) moves, the static scene's world point cloud
must stay put.

Quantitative check: voxelized overlap of each cam2 world cloud against
frame 0, and against the fixed cam1 cloud. The same metrics are also computed
for the three "flipped hand-eye" variants; the confirmed convention must win,
otherwise the convention is wrong and fusion must not start.

Outputs:
    output/debug/transform_validation.json
    output/debug/transforms/*.ply   (colored world clouds + cam2 trajectory)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.calibration.transforms import (
    invert_transform,
    pose7d_to_matrix,
)
from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.geometry.pointcloud import depth_to_pointcloud, transform_points
from object_reconstruction.utils.config import load_config
from object_reconstruction.utils.io import write_ply_points

VOXEL_SIZE_M = 0.02

VARIANTS = {
    "confirmed": (False, False),
    "flip_base_cam1": (True, False),
    "flip_tcp_cam2": (False, True),
    "flip_both": (True, True),
}


def voxel_set(points: np.ndarray) -> set[tuple[int, int, int]]:
    if len(points) == 0:
        return set()
    return set(map(tuple, np.floor(points / VOXEL_SIZE_M).astype(np.int64)))


def overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cam2_world_points(source, position, T_world_cam2, depth_cfg, stride, rgb=False):
    packet = source.read_packet(position)
    points, colors = depth_to_pointcloud(
        packet.cam2.depth,
        packet.cam2.intrinsics,
        depth_scale=depth_cfg["scale"],
        depth_min_m=depth_cfg["min_m"],
        depth_max_m=depth_cfg["max_m"],
        stride=stride,
        rgb=packet.cam2.rgb if rgb else None,
    )
    return transform_points(points, T_world_cam2), colors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--pixel-stride", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    depth_cfg = config["depth"]
    if depth_cfg.get("scale") is None:
        print("FAIL: depth.scale is null; confirm conventions before running Phase 1")
        return 1

    source = OfflineFrameSource.from_config(config)
    if not source.conventions_confirmed:
        print("FAIL: pose/handeye conventions are null; Phase 1 is blocked")
        return 1

    out_dir = Path(config["output"]["root"]) / "debug" / "transforms"
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = list(range(0, len(source), args.frame_stride))
    if positions[-1] != len(source) - 1:
        positions.append(len(source) - 1)

    # Fixed cam1 cloud (frame 0), WORLD = cam1 so no transform needed.
    packet0 = source.read_packet(0)
    cam1_points, cam1_colors = depth_to_pointcloud(
        packet0.cam1.depth,
        packet0.cam1.intrinsics,
        depth_scale=depth_cfg["scale"],
        depth_min_m=depth_cfg["min_m"],
        depth_max_m=depth_cfg["max_m"],
        stride=args.pixel_stride,
        rgb=packet0.cam1.rgb,
    )
    cam1_voxels = voxel_set(cam1_points)
    write_ply_points(out_dir / "cam1_world_000000.ply", cam1_points, cam1_colors)

    # Precompute per-frame T_base_tcp once; variants only flip hand-eye factors.
    T_base_tcp_all = {
        p: pose7d_to_matrix(source.raw_poses[p], source.quaternion_order) for p in positions
    }

    results: dict[str, dict] = {}
    trajectory = []
    for variant, (flip_base, flip_tcp) in VARIANTS.items():
        T_base_cam1 = invert_transform(source.T_base_cam1) if flip_base else source.T_base_cam1
        T_tcp_cam2 = invert_transform(source.T_tcp_cam2) if flip_tcp else source.T_tcp_cam2
        T_cam1_base = invert_transform(T_base_cam1)

        self_overlaps, cross_overlaps = [], []
        reference_voxels = None
        for p in positions:
            T_world_cam2 = T_cam1_base @ T_base_tcp_all[p] @ T_tcp_cam2
            world_points, colors = cam2_world_points(
                source, p, T_world_cam2, depth_cfg, args.pixel_stride,
                rgb=(variant == "confirmed" and p in (positions[0], positions[-1])),
            )
            voxels = voxel_set(world_points)
            if reference_voxels is None:
                reference_voxels = voxels
            else:
                self_overlaps.append(overlap(voxels, reference_voxels))
            cross_overlaps.append(overlap(voxels, cam1_voxels))

            if variant == "confirmed":
                trajectory.append(T_world_cam2[:3, 3])
                if colors is not None:
                    index = source.indices[p]
                    write_ply_points(
                        out_dir / f"cam2_world_{index:06d}.ply", world_points, colors
                    )

        results[variant] = {
            "cam2_static_overlap_mean": float(np.mean(self_overlaps)),
            "cam2_static_overlap_min": float(np.min(self_overlaps)),
            "cam2_vs_cam1_overlap_mean": float(np.mean(cross_overlaps)),
        }

    trajectory = np.asarray(trajectory)
    write_ply_points(
        out_dir / "cam2_trajectory.ply",
        trajectory,
        np.tile([255, 0, 0], (len(trajectory), 1)),
    )

    confirmed = results["confirmed"]
    best_static = max(results, key=lambda k: results[k]["cam2_static_overlap_mean"])
    best_cross = max(results, key=lambda k: results[k]["cam2_vs_cam1_overlap_mean"])
    # Criterion 1 (PLAN Phase 1 milestone): while cam2 moves, its world cloud
    # stays registered to itself. Validates pose semantics + T_tcp_cam2.
    static_passed = (
        best_static == "confirmed" and confirmed["cam2_static_overlap_mean"] >= 0.5
    )
    # Criterion 2 (required before dual-camera fusion): cam2 world cloud lands
    # on the fixed cam1 cloud. Additionally validates T_base_cam1.
    cross_passed = (
        best_cross == "confirmed" and confirmed["cam2_vs_cam1_overlap_mean"] >= 0.2
    )

    report = {
        "voxel_size_m": VOXEL_SIZE_M,
        "frames_evaluated": [source.indices[p] for p in positions],
        "results_per_variant": results,
        "best_variant_static": best_static,
        "best_variant_cross_camera": best_cross,
        "static_passed": static_passed,
        "cross_camera_passed": cross_passed,
        "passed": static_passed and cross_passed,
    }
    report_path = out_dir.parent / "transform_validation.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"frames evaluated: {len(positions)} (stride {args.frame_stride})")
    for variant, r in results.items():
        marker = " <= confirmed" if variant == "confirmed" else ""
        print(
            f"{variant:16s} static={r['cam2_static_overlap_mean']:.3f} "
            f"cross_cam={r['cam2_vs_cam1_overlap_mean']:.3f}{marker}"
        )
    print(f"cam2 trajectory span: {trajectory.min(axis=0)} .. {trajectory.max(axis=0)}")
    print(f"report: {report_path}")
    print(
        f"criterion 1 (cam2 static-scene consistency): "
        f"{'PASS' if static_passed else 'FAIL'} "
        f"({confirmed['cam2_static_overlap_mean']:.3f}, best={best_static})"
    )
    print(
        f"criterion 2 (cam2 vs cam1 cross-camera):     "
        f"{'PASS' if cross_passed else 'FAIL'} "
        f"({confirmed['cam2_vs_cam1_overlap_mean']:.3f}, best={best_cross})"
    )
    if static_passed and not cross_passed:
        print(
            "HINT: pose + T_tcp_cam2 are consistent but T_base_cam1 does not "
            "align cam1 with the scene; suspect the head hand-eye calibration "
            "(see output/debug/base_cam1_diagnosis.md). Dual-camera fusion is "
            "blocked; cam2-only reconstruction is unblocked."
        )
    return 0 if (static_passed and cross_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
