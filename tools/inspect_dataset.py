"""Phase 0 dataset inspection (PLAN section 6). No reconstruction here.

Checks every frame of the flat offline dataset, writes
output/debug/dataset_report.json, and saves a few side-by-side RGB/depth
preview montages. While pose/handeye/depth conventions are unconfirmed
(null in config), raw values are reported and blockers are listed;
T_world_cam2 is never constructed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.calibration.handeye import load_handeye_matrices
from object_reconstruction.data.offline_source import OfflineFrameSource
from object_reconstruction.utils.config import load_config

POSE_CROSSCHECK_TOLERANCE = 1e-4


def depth_to_visual(depth: np.ndarray) -> Image.Image:
    valid = depth[depth > 0]
    high = float(np.percentile(valid, 99)) if valid.size else 1.0
    scaled = np.clip(depth.astype(np.float64) / max(high, 1.0), 0.0, 1.0)
    return Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")


def save_preview(source: OfflineFrameSource, position: int, out_dir: Path) -> Path:
    packet = source.read_packet(position)
    tiles = [
        Image.fromarray(packet.cam1.rgb),
        depth_to_visual(packet.cam1.depth),
        Image.fromarray(packet.cam2.rgb),
        depth_to_visual(packet.cam2.depth),
    ]
    w, h = tiles[0].size
    montage = Image.new("RGB", (w * 2, h * 2))
    for i, tile in enumerate(tiles):
        montage.paste(tile, ((i % 2) * w, (i // 2) * h))
    montage = montage.resize((w, h))
    path = out_dir / f"preview_{packet.index:06d}.png"
    montage.save(path)
    return path


def inspect(config_path: str, preview_count: int) -> int:
    config = load_config(config_path)
    ds_config = config["dataset"]
    source = OfflineFrameSource.from_config(config)

    report: dict = {"dataset_root": str(source.root), "checks": {}, "blockers": []}
    failures: list[str] = []

    # Conventions gate.
    for key in ("pose_semantics", "quaternion_order", "handeye_convention"):
        if ds_config.get(key) is None:
            report["blockers"].append(f"dataset.{key} is null (confirm with capture side)")
    if config.get("depth", {}).get("scale") is None:
        report["blockers"].append("depth.scale is null (confirm 16-bit PNG unit)")

    # Completeness.
    report["checks"]["frame_count"] = len(source)
    report["checks"]["index_range"] = [source.indices[0], source.indices[-1]]
    report["checks"]["missing_files"] = {
        str(k): v for k, v in source.missing_files.items()
    }
    if source.missing_files:
        failures.append(f"{len(source.missing_files)} frame indices have missing files")

    # Intrinsics.
    for name, intr in (("cam1_head", source.intrinsics_cam1), ("cam2_wrist", source.intrinsics_cam2)):
        intr.validate()
        report["checks"][f"intrinsics_{name}"] = {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
        }

    # Hand-eye matrices: numeric sanity only; frame semantics stay unconfirmed.
    handeye = load_handeye_matrices(source.root / "handeye" / "handeye_tf.txt")
    report["checks"]["handeye"] = {}
    for label, T in handeye.items():
        det = float(np.linalg.det(T[:3, :3]))
        bottom_ok = bool(np.allclose(T[3], [0, 0, 0, 1]))
        report["checks"]["handeye"][label] = {"rotation_det": det, "bottom_row_ok": bottom_ok}
        if abs(det - 1.0) > 1e-3:
            failures.append(f"handeye {label}: det(R) = {det}")
        if not bottom_ok:
            failures.append(f"handeye {label}: bottom row is not [0,0,0,1]")

    # Raw pose table statistics + per-frame cross-check.
    poses = source.raw_poses
    report["checks"]["raw_pose_7d"] = {
        "rows": int(poses.shape[0]),
        "finite": bool(np.isfinite(poses).all()),
        "translation_min": poses[:, :3].min(axis=0).tolist(),
        "translation_max": poses[:, :3].max(axis=0).tolist(),
        "last4_norm_min": float(np.linalg.norm(poses[:, 3:], axis=1).min()),
        "last4_norm_max": float(np.linalg.norm(poses[:, 3:], axis=1).max()),
    }
    if not np.isfinite(poses).all():
        failures.append("raw pose table contains NaN/inf")

    max_pose_diff = 0.0
    for position, index in enumerate(source.indices):
        per_frame = source.load_per_frame_pose(index)
        max_pose_diff = max(max_pose_diff, float(np.abs(per_frame - poses[position]).max()))
    report["checks"]["pose_crosscheck_max_abs_diff"] = max_pose_diff
    if max_pose_diff > POSE_CROSSCHECK_TOLERANCE:
        failures.append(f"JointStates vs per-frame pose differ by {max_pose_diff}")

    # Per-frame image checks.
    shapes_ok = True
    depth_stats = []
    for position, index in enumerate(source.indices):
        packet = source.read_packet(position)
        for cam_name, cam in (("cam1", packet.cam1), ("cam2", packet.cam2)):
            expected = (cam.intrinsics.height, cam.intrinsics.width)
            if cam.rgb.shape != (*expected, 3) or cam.rgb.dtype != np.uint8:
                failures.append(f"frame {index} {cam_name}: bad RGB {cam.rgb.shape} {cam.rgb.dtype}")
                shapes_ok = False
            if cam.depth.shape != expected or cam.depth.dtype != np.uint16:
                failures.append(f"frame {index} {cam_name}: bad depth {cam.depth.shape} {cam.depth.dtype}")
                shapes_ok = False
            valid_ratio = float((cam.depth > 0).mean())
            depth_stats.append(
                {
                    "frame": index,
                    "cam": cam_name,
                    "min_raw": int(cam.depth[cam.depth > 0].min()) if (cam.depth > 0).any() else 0,
                    "max_raw": int(cam.depth.max()),
                    "valid_ratio": valid_ratio,
                }
            )
    report["checks"]["shapes_and_dtypes_ok"] = shapes_ok
    valid_ratios = [s["valid_ratio"] for s in depth_stats]
    report["checks"]["depth_summary"] = {
        "valid_ratio_min": min(valid_ratios),
        "valid_ratio_mean": float(np.mean(valid_ratios)),
        "max_raw_overall": max(s["max_raw"] for s in depth_stats),
    }
    report["depth_per_frame"] = depth_stats

    # Previews.
    out_root = Path(config["output"]["root"])
    debug_dir = out_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    preview_positions = np.linspace(0, len(source) - 1, num=preview_count, dtype=int)
    report["previews"] = [
        str(save_preview(source, int(p), debug_dir)) for p in dict.fromkeys(preview_positions)
    ]

    report["failures"] = failures
    report["passed"] = not failures
    report_path = debug_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"frames: {len(source)}  (indices {source.indices[0]}..{source.indices[-1]})")
    print(f"shapes/dtypes ok: {shapes_ok}")
    print(f"pose crosscheck max abs diff: {max_pose_diff:.2e}")
    print(f"depth valid ratio min/mean: {min(valid_ratios):.3f} / {np.mean(valid_ratios):.3f}")
    print(f"report: {report_path}")
    for blocker in report["blockers"]:
        print(f"BLOCKER: {blocker}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preview-frames", type=int, default=4)
    args = parser.parse_args()
    return inspect(args.config, args.preview_frames)


if __name__ == "__main__":
    raise SystemExit(main())
