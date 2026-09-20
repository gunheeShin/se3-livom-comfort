"""Offline odometry pipeline — runs SE3LIO over a dataset and collects a trajectory."""

import os
import time

import numpy as np

from se3_lio.se3_lio import SE3LIO


def _rot_to_quat_xyzw(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def _rss_mb():
    """Resident set size of this process in MB (/proc/self/statm, page-size units)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576
    except (OSError, ValueError):
        return float("nan")


class OdometryPipeline:
    """Run SE3LIO over an iterable dataset of frames and collect the trajectory."""

    def __init__(self, dataset, config, extrinsic=None, hba=None):
        self.dataset = dataset
        self.odometry = SE3LIO(config, extrinsic)
        self.hba = hba  # _OnlineHBA: gets every pose + deskewed cloud, finish() gives the refined poses
        self.poses_hba = None
        self.stamps = []
        self.poses = []  # list of 4x4
        self.pose_covs = []  # list of 6x6, error-state [t; omega] on T <- T*exp(xi)
        self.tracked = []  # visual points tracked per frame (0 when the camera is off)
        self.times_ms = []  # wall-clock of the core register call per frame
        self.rss_mb = []  # resident memory of this process after each frame (LIO + HBA worker)
        self.done = []  # time.perf_counter() when each pose came out (online-bag latency)

    def run(self, progress=True, logger=None, dump_dir=None):
        frames = self.dataset
        if progress:
            try:
                from tqdm import tqdm

                total = len(self.dataset) if hasattr(self.dataset, "__len__") else None
                frames = tqdm(self.dataset, total=total, desc="SE3-LIO", unit="frame")
            except ImportError:
                pass
        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)
        for frame in frames:
            t0 = time.perf_counter()
            if hasattr(frame, "scans"):
                state, cloud = self.odometry.register_multi(frame.scans, frame.imu, frame.end_time, frame.grays)
            else:
                state, cloud = self.odometry.register_frame(
                    frame.points, frame.point_times, frame.imu, frame.stamp, frame.lidar_idx,
                    frame.end_time, frame.grays,
                )
            self.done.append(time.perf_counter())
            self.times_ms.append((self.done[-1] - t0) * 1e3)
            self.rss_mb.append(_rss_mb())
            self.stamps.append(state.stamp)
            self.poses.append(np.array(state.pose))
            self.pose_covs.append(np.array(state.covariance)[:6, :6])
            self.tracked.append(self.odometry.num_tracked())
            if self.hba is not None:
                acc = frame.imu[:, 1:4].mean(0) if getattr(frame, "imu", None) is not None and len(frame.imu) else np.full(3, np.nan)
                self.hba.push(_rot_to_quat_xyzw(self.poses[-1][:3, :3]), self.poses[-1][:3, 3], cloud, float(state.stamp),
                              acc, np.asarray(state.ba, dtype=float), np.asarray(state.grav, dtype=float))
            if dump_dir is not None:
                _write_pcd(os.path.join(dump_dir, f"{len(self.poses) - 1:05d}.pcd"), cloud)
            if logger is not None:
                logger.log_frame(state.stamp, self.poses[-1], getattr(frame, "points", cloud), state.grav)  # multi: deskewed body-frame cloud
        if self.hba is not None:
            self.poses_hba = list(self.hba.finish())
        return self

    def save_tum(self, path, poses=None):
        with open(path, "w") as f:
            for t, T in zip(self.stamps, self.poses if poses is None else poses):
                p = T[:3, 3]
                q = _rot_to_quat_xyzw(T[:3, :3])
                f.write(
                    f"{t:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
                )

    def save_timing(self, path):
        with open(path, "w") as f:
            f.write("stamp,ms,rss_mb\n")
            for t, ms, mb in zip(self.stamps, self.times_ms, self.rss_mb):
                f.write(f"{t:.9f},{ms:.3f},{mb:.0f}\n")

    def save_cov(self, path):
        np.save(path, np.asarray(self.pose_covs))

    def path_length(self):
        if len(self.poses) < 2:
            return 0.0
        pos = np.array([T[:3, 3] for T in self.poses])
        return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())

    def summary(self):
        n = len(self.poses)
        if n == 0:
            return "no frames processed"
        final = self.poses[-1][:3, 3]
        return (
            f"frames: {n}\n"
            f"path length: {self.path_length():.3f} m\n"
            f"final position: [{final[0]:.3f}, {final[1]:.3f}, {final[2]:.3f}]"
            + (f"\nvisual tracked/frame: median {np.median(self.tracked):.0f}, min {min(self.tracked)}"
               if any(self.tracked) else "")
            + f"\ncore ms/frame: mean {np.mean(self.times_ms):.1f}, p95 {np.percentile(self.times_ms, 95):.1f}, "
              f"max {np.max(self.times_ms):.1f}, >100ms {np.mean(np.array(self.times_ms) > 100) * 100:.2f}%"
        )


def _write_pcd(path, xyz):
    """Deskewed body-frame scan as binary PCD (x y z intensity), one file per trajectory row (HBA input)."""
    n = len(xyz)
    data = np.zeros((n, 4), dtype=np.float32)
    data[:, :3] = xyz
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
              f"TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode()); f.write(data.tobytes())
