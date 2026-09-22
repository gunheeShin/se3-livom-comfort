"""High-level Python API over the raw pybind module."""

import numpy as np

from se3_lio.pybind import se3_lio_pybind as _pb
from se3_lio.config import SE3LIOConfig


def _grays(imgs):
    return [] if imgs is None else [np.empty((0, 0), np.uint8) if g is None else np.ascontiguousarray(g, dtype=np.uint8) for g in imgs]


class SE3LIO:
    """SE(3) LiDAR-Inertial Odometry.

    Wraps the C++ core. `register_frame` reproduces the ROS2 node's per-frame
    data path (build measurement -> apply LiDAR extrinsic -> sort points by
    relative timestamp -> estimatePose) and returns the updated state.
    """

    def __init__(self, config: SE3LIOConfig = None, lidar_extrinsic=None):
        self.config = config if config is not None else SE3LIOConfig()
        self.extrinsic = np.eye(4) if lidar_extrinsic is None else np.asarray(
            lidar_extrinsic, dtype=float
        )
        self._odom = _pb._SE3LIO(self.config.to_pybind(), self.extrinsic)

    def register_frame(self, points, point_times, imu, frame_stamp, lidar_idx=None, end_time=None, grays=None):
        """Run one odometry step.

        points:      (N, 3) float xyz in the LiDAR frame
        point_times: (N,)   float per-point time offset from frame start [s]
        imu:         (M, 7) float rows of [t, ax, ay, az, gx, gy, gz]
        frame_stamp: float, absolute start time of the scan [s]
        lidar_idx:   (N,) int per-point LiDAR index into lidar_*_noises (None -> 0)
        end_time:    float, absolute epoch end the state is propagated to (None -> last point)
        grays:       list of (H, W) uint8 undistorted images at end_time, one per config camera
                     (None / empty entry -> no photometric update for that camera)
        Returns ``(state, cloud)``:
          state: pybind _State (pose, vel, bg, ba, grav, covariance, ...)
          cloud: (N, 3) deskewed scan points in the body frame (post-extrinsic).
        """
        return self._odom._register_frame(
            np.ascontiguousarray(points, dtype=float),
            np.ascontiguousarray(point_times, dtype=float).ravel(),
            np.ascontiguousarray(imu, dtype=float),
            float(frame_stamp),
            np.empty(0) if lidar_idx is None else np.ascontiguousarray(lidar_idx, dtype=float).ravel(),
            0.0 if end_time is None else float(end_time),
            _grays(grays),
        )

    def register_multi(self, scans, imu, end_time=None, grays=None):
        """One odometry step from several LiDARs merged inside the core.

        scans: list of (points (N_i,3) in sensor i frame, point_times (N_i,), stamp) — one per
               LiDAR, index i matching config.lidar_extrinsics / lidar_*_noises.
        imu:   (M, 7) rows of [t, ax, ay, az, gx, gy, gz]
        end_time: absolute epoch end (None -> last merged point)
        grays:    list of (H, W) uint8 undistorted images at end_time, one per config camera
        Returns (state, cloud) like register_frame; cloud is the deskewed merged scan.
        """
        pts = [np.ascontiguousarray(p, dtype=float) for p, _, _ in scans]
        tms = [np.ascontiguousarray(t, dtype=float).ravel() for _, t, _ in scans]
        stamps = [float(s) for _, _, s in scans]
        return self._odom._register_multi(pts, tms, stamps, np.ascontiguousarray(imu, dtype=float),
                                          0.0 if end_time is None else float(end_time), _grays(grays))

    def num_tracked(self):
        """Visual map points used by the last photometric update (0 when the camera is off)."""
        return self._odom.num_tracked()

    def leaf(self):
        """Downsample grid (m) of the last scan; moves with the inlier count when target_inliers > 0."""
        return self._odom.leaf()

    def inliers(self):
        """LiDAR inliers of the last update (the count the adaptive grid reacts to)."""
        return self._odom.inliers()

    def merge_lidars(self, scans):
        """Merge only (no estimation): returns ((N,5) [x y z lidar_idx t_offset] in body frame, stamp)."""
        pts = [np.ascontiguousarray(p, dtype=float) for p, _, _ in scans]
        tms = [np.ascontiguousarray(t, dtype=float).ravel() for _, t, _ in scans]
        return self._odom._merge_lidars(pts, tms, [float(s) for _, _, s in scans])
