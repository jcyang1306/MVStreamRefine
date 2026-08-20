"""Incremental reconstruction viewer (PLAN section 14, Task 8).

Shows the TSDF-extracted object point cloud, the fixed cam1 (WORLD) frame,
the current cam2 frame and the accumulated cam2 trajectory. Refreshed by the
pipeline every N accepted keyframes; never per input frame.

Hotkeys:
    SPACE  pause / resume (the loop blocks while paused, window stays live)
    S      save current point cloud to output/pointcloud/
    M      save current mesh to output/mesh/
    Q      quit (the pipeline stops early)

open3d is imported lazily; on machines without a display run the pipeline
with the viewer disabled (visualization.enabled: false or --no-viewer).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

_KEY_SPACE, _KEY_S, _KEY_M, _KEY_Q = 32, 83, 77, 81


class NullViewer:
    """No-op stand-in used when visualization is disabled or headless."""

    def update(self, point_cloud: Any, T_world_cam2: np.ndarray) -> bool:
        return True

    def close(self) -> None:
        pass


class LiveViewer:
    def __init__(
        self,
        on_save_point_cloud: Callable[[], Any] | None = None,
        on_save_mesh: Callable[[], Any] | None = None,
        window_name: str = "MVStreamRefine - object A",
        frame_size_m: float = 0.05,
    ) -> None:
        try:
            import open3d
        except ImportError as exc:
            raise RuntimeError(
                "open3d is required for the live viewer; run inside the CUDA "
                "container or disable it (--no-viewer / visualization.enabled: false)"
            ) from exc
        self._o3d = open3d
        self._on_save_point_cloud = on_save_point_cloud
        self._on_save_mesh = on_save_mesh
        self._paused = False
        self._quit = False

        self._vis = open3d.visualization.VisualizerWithKeyCallback()
        if not self._vis.create_window(window_name=window_name, width=1280, height=720):
            raise RuntimeError(
                "could not create the Open3D window. When running in Docker, "
                "pass DISPLAY and mount /tmp/.X11-unix, or run with --no-viewer."
            )
        self._vis.register_key_callback(_KEY_SPACE, self._toggle_pause)
        self._vis.register_key_callback(_KEY_S, self._save_point_cloud)
        self._vis.register_key_callback(_KEY_M, self._save_mesh)
        self._vis.register_key_callback(_KEY_Q, self._request_quit)

        self._pcd = open3d.geometry.PointCloud()
        self._trajectory = open3d.geometry.PointCloud()
        self._cam2_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size_m
        )
        self._T_cam2_shown = np.eye(4)
        # cam1 == WORLD: a static frame at the origin.
        self._vis.add_geometry(
            open3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size_m)
        )
        self._vis.add_geometry(self._pcd)
        self._vis.add_geometry(self._trajectory)
        self._vis.add_geometry(self._cam2_frame)
        self._first_update = True

    # -- key callbacks -------------------------------------------------------

    def _toggle_pause(self, _vis) -> bool:
        self._paused = not self._paused
        return False

    def _request_quit(self, _vis) -> bool:
        self._quit = True
        self._paused = False
        return False

    def _save_point_cloud(self, _vis) -> bool:
        if self._on_save_point_cloud is not None:
            self._on_save_point_cloud()
        return False

    def _save_mesh(self, _vis) -> bool:
        if self._on_save_mesh is not None:
            self._on_save_mesh()
        return False

    # -- pipeline interface ----------------------------------------------------

    def update(self, point_cloud: Any, T_world_cam2: np.ndarray) -> bool:
        """Refresh all geometry; returns False when the user requested quit."""
        legacy = point_cloud.to_legacy() if hasattr(point_cloud, "to_legacy") else point_cloud
        self._pcd.points = legacy.points
        self._pcd.colors = legacy.colors

        T = np.asarray(T_world_cam2, dtype=np.float64)
        self._cam2_frame.transform(T @ np.linalg.inv(self._T_cam2_shown))
        self._T_cam2_shown = T
        self._trajectory.points.append(T[:3, 3])
        self._trajectory.paint_uniform_color([1.0, 0.0, 0.0])

        self._vis.update_geometry(self._pcd)
        self._vis.update_geometry(self._trajectory)
        self._vis.update_geometry(self._cam2_frame)
        if self._first_update:
            self._vis.reset_view_point(True)
            self._first_update = False

        self._pump()
        while self._paused and not self._quit:
            self._pump()
        return not self._quit

    def _pump(self) -> None:
        self._vis.poll_events()
        self._vis.update_renderer()

    def close(self) -> None:
        self._vis.destroy_window()
