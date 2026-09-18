"""Rerun logging for SE(3)-LIO.

Each keyframe's LiDAR scan is transformed into the world frame and logged as a
point cloud (``window=N`` keeps only the last N -- a sliding submap); the
trajectory and sensor pose are logged every frame, and per-frame linear/angular
speed + CPU/RAM/compute-time are logged as scalar plots. The world is rotated so
gravity points down (z up). A blueprint bakes the layout (3D view that follows
the sensor + a row of plots), point/line style defaults, and the sensor TF axes,
all overridable in the Rerun UI.
"""

import os
import time

import numpy as np

from se3_lio.viz._geometry import lidar_to_world, gravity_align


def _rss_mb():
    """Current process resident memory in MB (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _set_time(rr, timeline, seconds):
    """Set the active time, across rerun API versions (<=0.22 vs >=0.23)."""
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(timeline, seconds)
    else:
        rr.set_time(timeline, timestamp=seconds)


class RerunLogger:
    """Logs trajectory + per-keyframe world scan into Rerun. By default scans
    accumulate into the full map; pass ``window=N`` to keep only the last N (a
    sliding submap) so the viewer stays light. The trajectory always stays whole."""

    def __init__(self, extrinsic, app_id="se3_lio", keyframe_dist=0.0, save_path=None,
                 window=0, axis_length=1.5):
        import rerun as rr

        self._rr = rr
        self._extrinsic = np.asarray(extrinsic, dtype=float)
        self._keyframe_dist = float(keyframe_dist)
        # Scan-map window: 0 keeps every keyframe (whole accumulated map); N reuses
        # a ring of N entity paths so only the last N keyframe scans stay visible (a
        # sliding submap -- lighter viewer). The trajectory always accumulates fully.
        self._window = int(window)
        self._axis_length = float(axis_length)
        self._last_kf_pos = None
        self._traj = []
        self._R = None  # gravity-alignment rotation, fixed from the first frame
        self._i = 0
        self._prev_pose = None   # (stamp, pos, R) -> linear/angular speed
        self._prev_cpu = None    # (cpu_seconds, wall) -> CPU%
        self._t_prev_end = None  # wall at end of last log_frame -> per-frame compute time
        rr.init(app_id)
        # With a save_path, attach the file sink up front so each log streams to
        # disk instead of accumulating the whole recording in memory.
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            rr.save(save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # Legend names (+ units) for the scalar plots.
        rr.log("velocity/linear", rr.SeriesLines(names="linear velocity (m/s)"), static=True)
        rr.log("velocity/angular", rr.SeriesLines(names="angular velocity (deg/s)"), static=True)
        rr.log("system/cpu", rr.SeriesLines(names="CPU (%)"), static=True)
        rr.log("system/ram", rr.SeriesLines(names="RAM (MB)"), static=True)
        rr.log("system/compute", rr.SeriesLines(names="compute time (ms)"), static=True)
        self._send_blueprint()

    def _send_blueprint(self):
        """Bake the default look + a camera that follows the sensor into the
        recording, so it opens tracking `world/sensor` (camera-follow) with a dark
        background and point defaults -- all overridable in the Rerun UI."""
        import rerun.blueprint as rrb

        rr = self._rr
        view3d = rrb.Spatial3DView(
            origin="/world",
            background=[18, 18, 26],
            eye_controls=rrb.EyeControls3D(tracking_entity="/world/sensor"),
            # Style lives here as view defaults (map points, trajectory lines), so
            # it stays editable in the viewer instead of baked per-frame.
            defaults=[
                rr.Points3D.from_fields(colors=[200, 200, 200], radii=0.05),
                rr.LineStrips3D.from_fields(colors=[255, 180, 40], radii=0.05),
            ],
            # The sensor TF axes are an override (not logged data), so axis length
            # is editable in the viewer; they also give world/sensor the bounding
            # box the camera-follow needs to center on.
            overrides={"world/sensor": [rr.TransformAxes3D(axis_length=self._axis_length)]},
        )
        # 3D view on top, the scalar plots (speed + compute/CPU/RAM) in a row below.
        rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
            view3d,
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/velocity/linear", name="linear velocity (m/s)"),
                rrb.TimeSeriesView(origin="/velocity/angular", name="angular velocity (deg/s)"),
                rrb.TimeSeriesView(origin="/system/compute", name="compute time (ms)"),
                rrb.TimeSeriesView(origin="/system/cpu", name="CPU (%)"),
                rrb.TimeSeriesView(origin="/system/ram", name="RAM (MB)"),
            ),
            row_shares=[3, 1],
        )))

    def log_frame(self, stamp, pose, scan_pts, grav=None):
        rr = self._rr
        # Time since the previous log_frame returned ~= the pipeline's compute for
        # this frame (excludes this call's own logging, which runs after).
        t_start = time.monotonic()
        pose = np.asarray(pose, dtype=float)
        # Fix the gravity-alignment rotation from the first frame, then apply it
        # to every pose and scan so the streamed world has gravity along -z.
        if self._R is None:
            self._R = gravity_align(grav)
        R = self._R
        pos = R @ pose[:3, 3]

        _set_time(rr, "stamp", float(stamp))
        if self._t_prev_end is not None:
            rr.log("system/compute", rr.Scalars((t_start - self._t_prev_end) * 1000.0))

        # A keyframe is the first frame, or once the sensor has moved
        # >= keyframe_dist since the last one (keyframe_dist <= 0 keeps all).
        moved = (
            self._last_kf_pos is None
            or float(np.linalg.norm(pos - self._last_kf_pos)) >= self._keyframe_dist
        )
        is_keyframe = self._keyframe_dist <= 0.0 or moved

        if is_keyframe:  # skip empty scans (possible after range/tag filtering)
            world = lidar_to_world(pose, self._extrinsic, scan_pts) @ R.T
            if world.shape[0] > 0:
                self._last_kf_pos = pos.copy()
                slot = self._i % self._window if self._window > 0 else self._i
                rr.log(f"world/map/{slot:05d}", rr.Points3D(world))
                self._i += 1

        # Grow the path one short segment at a time. Re-logging the whole strip
        # every frame would re-store the full path each frame (O(n^2)); a 2-point
        # segment per frame under its own path keeps storage O(n). Style comes from
        # the view's LineStrips3D default (see _send_blueprint), editable in the UI.
        if self._traj:
            rr.log(
                f"world/trajectory/{len(self._traj):05d}",
                rr.LineStrips3D([np.array([self._traj[-1], pos])]),  # style from view default
            )
        self._traj.append(pos)
        # Pose only; the TF axes are drawn by the blueprint override (see above).
        rr.log("world/sensor", rr.Transform3D(translation=pos, mat3x3=R @ pose[:3, :3]))

        # Scalar plots: linear/angular speed (from consecutive poses) + CPU/RAM.
        if self._prev_pose is not None:
            pt, pp, pR = self._prev_pose
            dt = float(stamp) - pt
            if dt > 0:
                rr.log("velocity/linear", rr.Scalars(float(np.linalg.norm(pos - pp) / dt)))
                dR = pR.T @ (pose[:3, :3])
                ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))) / dt
                rr.log("velocity/angular", rr.Scalars(float(ang)))
        self._prev_pose = (float(stamp), pos.copy(), pose[:3, :3].copy())

        cpu_s = sum(os.times()[:2])   # process user+system CPU seconds
        wall = time.monotonic()
        if self._prev_cpu is not None:
            dw = wall - self._prev_cpu[1]
            if dw > 0:
                rr.log("system/cpu", rr.Scalars(100.0 * (cpu_s - self._prev_cpu[0]) / dw))
        self._prev_cpu = (cpu_s, wall)
        rr.log("system/ram", rr.Scalars(_rss_mb()))
        self._t_prev_end = time.monotonic()

    def save(self, path):
        self._rr.save(path)


def read_rrd_trajectory(path):
    """Read back the logged sensor trajectory from a .rrd recording.

    Returns (stamps (M,), positions (M, 3)). Used by the G2 smoke test and the
    G3 check to compare the recording against the estimated trajectory.
    """
    import pyarrow as pa
    import rerun as rr

    recording = rr.dataframe.load_recording(path)
    view = recording.view(index="stamp", contents={"world/sensor": ["Translation3D"]})
    table = view.select().read_all()

    stamp_col = table.column("stamp").combine_chunks()
    if pa.types.is_timestamp(stamp_col.type):
        stamps = stamp_col.cast(pa.int64()).to_numpy() / 1e9
    else:
        stamps = np.asarray(stamp_col.to_numpy(zero_copy_only=False), dtype=float)

    col = table.column("/world/sensor:Translation3D").to_pylist()
    positions = np.array([np.asarray(v, dtype=float).reshape(-1)[:3] for v in col])
    return stamps, positions
