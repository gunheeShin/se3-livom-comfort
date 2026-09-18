"""Live Polyscope viewer for SE(3)-LIO (optional, real-time).

A lightweight interactive visualizer: the current scan (in the gravity-aligned
world frame) and the trajectory, with play/pause/step controls. Polyscope is an
optional dependency imported lazily — install with ``pip install se3-lio[viz]``.
Follows the Polyscope visualizers in KISS-ICP / GenZ-ICP.
"""

import datetime
import importlib

import numpy as np

from se3_lio.viz._geometry import lidar_to_world, gravity_align

START_BUTTON = " START\n[SPACE]"
PAUSE_BUTTON = " PAUSE\n[SPACE]"
NEXT_BUTTON = "NEXT\n [N]"
CENTER_BUTTON = "CENTER\n  [C]"
SCREENSHOT_BUTTON = "SHOT\n [S]"

BACKGROUND_COLOR = [0.07, 0.07, 0.10]
FRAME_COLOR = [0.85, 0.11, 0.38]
TRAJECTORY_COLOR = [0.12, 0.53, 0.90]
FRAME_PTS_SIZE = 0.05


class PolyscopeVisualizer:
    """Real-time viewer with the pipeline logger interface (``log_frame``)."""

    def __init__(self, extrinsic):
        try:
            self._ps = importlib.import_module("polyscope")
            self._gui = self._ps.imgui
        except ModuleNotFoundError:
            print('polyscope is not installed — run "pip install se3-lio[viz]"')
            raise SystemExit(1)

        self._extrinsic = np.asarray(extrinsic, float)
        self._R = None  # gravity-alignment rotation, fixed from the first frame
        self._trajectory = []
        self._frame_size = FRAME_PTS_SIZE
        self._block = True
        self._play = False

        self._ps.set_program_name("SE(3)-LIO Visualizer")
        self._ps.init()
        self._ps.set_up_dir("z_up")
        self._ps.set_ground_plane_mode("none")
        self._ps.set_background_color(BACKGROUND_COLOR)
        self._ps.set_verbosity(0)
        self._ps.set_build_default_gui_panels(False)
        self._ps.set_user_callback(self._gui_callback)

    def log_frame(self, stamp, pose, scan_pts, grav=None):
        pose = np.asarray(pose, float)
        if self._R is None:
            self._R = gravity_align(grav)
        R = self._R
        world = lidar_to_world(pose, self._extrinsic, scan_pts) @ R.T
        self._trajectory.append(R @ pose[:3, 3])
        self._draw(world)

        # Step/play loop (KISS-ICP style): block per frame until NEXT/PLAY.
        while self._block:
            self._ps.frame_tick()
            if self._play:
                break
        self._block = not self._block

    def hold(self):
        """Keep the window open after the run so the final state can be inspected."""
        self._play = False
        self._ps.show()

    def _draw(self, world):
        if world.shape[0] > 0:
            cloud = self._ps.register_point_cloud(
                "current_frame", world, color=FRAME_COLOR, point_render_mode="quad"
            )
            cloud.set_radius(self._frame_size, relative=False)
        traj = np.asarray(self._trajectory)
        if len(traj) >= 2:
            edges = np.column_stack([np.arange(len(traj) - 1), np.arange(1, len(traj))])
            curve = self._ps.register_curve_network(
                "trajectory", traj, edges, color=TRAJECTORY_COLOR
            )
            curve.set_radius(0.04, relative=False)

    def _gui_callback(self):
        name = PAUSE_BUTTON if self._play else START_BUTTON
        if self._gui.Button(name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Space):
            self._play = not self._play
        self._gui.SameLine()
        if self._gui.Button(NEXT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self._block = not self._block
        self._gui.SameLine()
        if self._gui.Button(CENTER_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_C):
            self._ps.reset_camera_to_home_view()
        self._gui.SameLine()
        if self._gui.Button(SCREENSHOT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_S):
            self._ps.screenshot(
                "se3lio_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
            )
        changed, self._frame_size = self._gui.SliderFloat(
            "frame size", self._frame_size, v_min=0.01, v_max=0.4
        )
        if changed and self._ps.has_point_cloud("current_frame"):
            self._ps.get_point_cloud("current_frame").set_radius(self._frame_size, relative=False)
