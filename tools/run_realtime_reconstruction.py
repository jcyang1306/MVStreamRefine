"""Realtime single wrist-camera object reconstruction (RealSense + RealMan arm).

Live loop: synchronized RGB-D + robot pose packets are shown in an OpenCV
window; after an ROI is selected, SAM 2.1 tracks the object frame by frame
and every accepted keyframe is fused into the shared TSDF engine (optional
ICP refinement). The Open3D viewer shows the growing model.

OpenCV window hotkeys:
    B  select / reselect the object ROI (drag, Enter/Space confirm)
    R  confirm the mask and start integrating / resume
    P  pause integration (tracking continues)
    C  clear tracking, keep the model            N  start a new model
    S  save a point-cloud snapshot               Q / ESC  quit and export

Requires hardware + CUDA stack (pyrealsense2, Robotic_Arm, torch/sam2,
open3d, GUI opencv), so run inside the container with X11 and devices:
    docker compose run --rm reconstruction \
        python3 tools/run_realtime_reconstruction.py --config configs/realtime.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from object_reconstruction.data.realsense_source import RealSenseSource
from object_reconstruction.data.realtime_source import RealtimeFrameSource
from object_reconstruction.data.robot_pose_source import (
    RobotPosePoller,
    connect_realman_arm,
)
from object_reconstruction.fusion.keyframe_selector import KeyframeSelector
from object_reconstruction.pipeline.realtime_pipeline import RealtimePipeline, State
from object_reconstruction.pipeline.reconstruction_engine import ReconstructionEngine
from object_reconstruction.segmentation.sam2_stream_segmenter import SAM2StreamSegmenter
from object_reconstruction.utils.config import load_config
from object_reconstruction.utils.logging import setup_logging

logger = logging.getLogger("tools.run_realtime_reconstruction")

WINDOW = "MVStreamRefine realtime - object A"

_STATE_COLORS = {
    State.PREVIEW: (200, 200, 200),
    State.MASK_CONFIRM: (0, 255, 255),
    State.RUNNING: (0, 255, 0),
    State.PAUSED: (0, 165, 255),
    State.LOST: (0, 0, 255),
}


def make_engine_factory(config: dict, refiner_enabled: bool):
    from object_reconstruction.fusion.tsdf_volume import TSDFVolume

    def factory() -> ReconstructionEngine:
        refiner = None
        if refiner_enabled:
            from object_reconstruction.registration.icp_refiner import ICPRefiner
            refiner = ICPRefiner.from_config(config)
        return ReconstructionEngine(
            TSDFVolume.from_config(config),
            KeyframeSelector.from_config(config),
            config,
            icp_refiner=refiner,
        )

    return factory


def build_viewer(config: dict, output_root: Path, no_viewer: bool, pipeline):
    if no_viewer or not bool(config.get("visualization", {}).get("enabled", False)):
        from object_reconstruction.visualization.live_viewer import NullViewer
        return NullViewer()
    from object_reconstruction.visualization.live_viewer import LiveViewer

    def save_point_cloud() -> None:
        path = output_root / "pointcloud" / f"model_{datetime.now():%H%M%S}.ply"
        pipeline.engine.tsdf.save_point_cloud(path)
        logger.info("hotkey save: %s", path)

    def save_mesh() -> None:
        path = output_root / "mesh" / f"model_{datetime.now():%H%M%S}.ply"
        pipeline.engine.tsdf.save_mesh(path)
        logger.info("hotkey save: %s", path)

    return LiveViewer(on_save_point_cloud=save_point_cloud, on_save_mesh=save_mesh,
                      window_name="MVStreamRefine realtime - model")


def draw_overlay(cv2, packet, output, source, pipeline, fps: float, message: str):
    import numpy as np

    bgr = np.ascontiguousarray(packet.cam.rgb[:, :, ::-1])
    if output.mask is not None:
        bgr[output.mask] = (0.5 * bgr[output.mask]
                            + 0.5 * np.array([0.0, 0.0, 255.0])).astype(np.uint8)

    color = _STATE_COLORS[output.state]
    sync = (f"{source.last_sync_error_ms:.0f}ms"
            if source.last_sync_error_ms is not None else "-")
    lines = [
        f"{output.state.value}  kf={pipeline.engine.keyframes}  "
        f"mask={output.mask_area_px}px  fps={fps:.1f}",
        f"sync={sync}  dropped={source.frames_dropped_sync}  "
        f"pose_err={source.pose_poller.read_failures}",
        "B roi  R run  P pause  C clear  N new  S save  Q quit",
    ]
    if message:
        lines.append(message)
    for i, line in enumerate(lines):
        cv2.putText(bgr, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    return bgr


def select_roi(cv2, packet) -> tuple[float, float, float, float] | None:
    import numpy as np

    bgr = np.ascontiguousarray(packet.cam.rgb[:, :, ::-1])
    x, y, w, h = cv2.selectROI(WINDOW, bgr, showCrosshair=True, fromCenter=False)
    if w <= 0 or h <= 0:
        return None
    return (float(x), float(y), float(x + w), float(y + h))


def export(pipeline, output_root: Path, debug_records: list[dict], save_debug: bool):
    tsdf = pipeline.engine.tsdf
    if pipeline.engine.keyframes == 0:
        logger.warning("no keyframe was integrated; nothing to export")
        return
    pc_path = output_root / "pointcloud" / "object_a.ply"
    tsdf.save_point_cloud(pc_path)
    mesh_path = output_root / "mesh" / "object_a_mesh.ply"
    tsdf.save_mesh(mesh_path)
    logger.info("saved point cloud: %s", pc_path)
    logger.info("saved mesh:        %s", mesh_path)
    if save_debug and debug_records:
        debug_path = output_root / "debug" / "fusion_debug.jsonl"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8") as fh:
            for record in debug_records:
                fh.write(json.dumps(record) + "\n")
        logger.info("debug records: %s (%d keyframes)", debug_path, len(debug_records))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-viewer", action="store_true",
                        help="disable the Open3D model viewer (OpenCV window stays)")
    parser.add_argument("--icp", choices=["true", "false"], default=None,
                        help="override icp.enabled")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(config["output"]["root"])
    log_path = setup_logging(output_root)
    logger.info("log file: %s", log_path)

    icp_enabled = bool(config.get("icp", {}).get("enabled", False))
    if args.icp is not None:
        icp_enabled = args.icp == "true"
    logger.info("icp: %s", "on" if icp_enabled else "off (robot pose only)")

    import cv2

    robot_cfg = config["realtime"]["robot"]
    read_pose = connect_realman_arm(robot_cfg["ip"], int(robot_cfg["port"]))
    poller = RobotPosePoller.from_config(config, read_pose)
    camera = RealSenseSource.from_config(config)
    source = RealtimeFrameSource.from_config(config, camera, poller)

    pipeline = RealtimePipeline(
        engine_factory=make_engine_factory(config, icp_enabled),
        segmenter=SAM2StreamSegmenter.from_config(config),
        config=config,
    )
    viewer = build_viewer(config, output_root, args.no_viewer, pipeline)
    update_every = max(1, int(config.get("visualization", {})
                              .get("update_every_keyframes", 2)))
    save_debug = bool(config["output"].get("save_debug", False))

    debug_records: list[dict] = []
    shown_keyframes = 0
    fps, fps_t0, fps_n = 0.0, time.monotonic(), 0
    message = ""

    source.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    logger.info("realtime loop started; press B to select the object")
    try:
        while True:
            packet = source.read_packet()
            if not viewer.poll():
                break
            if packet is None:
                message = "sync drop (no robot pose close enough)"
                # Keep the UI responsive (and quittable) during sync outages.
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            output = pipeline.process(packet)
            if output.message:
                message = output.message
            result = output.engine_result
            if result is not None and result.integrated:
                debug_records.append(result.debug_record)
                logger.info("[frame %4d] keyframe #%d  mask_area=%d",
                            packet.index, result.keyframe_id,
                            result.rgbd.mask_area_px)
                if pipeline.engine.keyframes - shown_keyframes >= update_every:
                    shown_keyframes = pipeline.engine.keyframes
                    if not viewer.update(pipeline.engine.tsdf.extract_point_cloud(),
                                         packet.T_world_cam):
                        break

            fps_n += 1
            if (now := time.monotonic()) - fps_t0 >= 1.0:
                fps, fps_t0, fps_n = fps_n / (now - fps_t0), now, 0

            cv2.imshow(WINDOW, draw_overlay(cv2, packet, output, source,
                                            pipeline, fps, message))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                box = select_roi(cv2, packet)
                message = ("ROI cancelled" if box is None else
                           f"ROI {tuple(int(v) for v in box)}; confirm with R")
                if box is not None:
                    pipeline.set_roi(packet.cam.rgb, box)
            elif key == ord("s"):
                if pipeline.engine.keyframes > 0:
                    path = (output_root / "pointcloud"
                            / f"model_{datetime.now():%H%M%S}.ply")
                    pipeline.engine.tsdf.save_point_cloud(path)
                    message = f"saved {path.name}"
            elif key != 255:
                if (msg := pipeline.handle_key(chr(key))) is not None:
                    message = msg
                    logger.info("%s", msg)
    finally:
        source.stop()
        cv2.destroyAllWindows()
        viewer.close()

    logger.info("frames seen: %d  keyframes: %d  sync drops: %d",
                pipeline.frames_seen, pipeline.engine.keyframes,
                source.frames_dropped_sync)
    export(pipeline, output_root, debug_records, save_debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
