"""ROS2 rosbag dataset for SE(3)-LIO.

Reproduces the ROS2 node's data path exactly so a Python run matches the node:
  - ros2_conversion.h::convertIMUMessage / convertLivoxMessage
  - MeasurementSynchronizer::synchronizeIMULiDAR

Reading uses rosbag2_py + the message Python modules, so it must run in an
environment where ROS2 (and livox_ros_driver2) is sourced — the same env the
binding is built in.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    points: np.ndarray  # (N, 3) xyz in the LiDAR frame
    point_times: np.ndarray  # (N,) per-point time offset from frame start [s]
    imu: np.ndarray  # (M, 7) rows of [t, ax, ay, az, gx, gy, gz]
    stamp: float  # absolute scan start time [s]
    lidar_idx: np.ndarray = None  # (N,) per-point LiDAR index for merged scans (None -> 0)
    end_time: float = None  # absolute epoch end the state is propagated to (None -> last point)
    grays: list = None  # per-camera (H, W) uint8 undistorted images at end_time (None -> LIO only)


def _open_reader(bag_path):
    import rosbag2_py

    storage = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    conv = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, conv)
    return reader


def _convert_livox(msg, min_range, point_filter_num=1):
    """Port of convertLivoxMessage: returns (pts Nx3 float64, offsets N float64)."""
    pts = msg.points
    n = len(pts)
    min_r2 = min_range * min_range
    last_x = last_y = last_z = 0.0
    xs, ys, zs, offs = [], [], [], []
    for i in range(0, n, point_filter_num):  # keep the first point too (mirrors the node's i=0 fix)
        pt = pts[i]
        tag = pt.tag & 0x30
        if tag != 0x10 and tag != 0x00:
            continue
        x, y, z = pt.x, pt.y, pt.z
        moved = (
            abs(x - last_x) > 1e-7 or abs(y - last_y) > 1e-7 or abs(z - last_z) > 1e-7
        )
        if moved and (x * x + y * y + z * z > min_r2):
            offset = pt.offset_time * 1e-9
            if offset > 0.1:
                continue  # skip; last_* is NOT updated (matches C++ `continue`)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            offs.append(offset)
        last_x, last_y, last_z = x, y, z
    pts_arr = np.array([xs, ys, zs], dtype=np.float64).T.reshape(-1, 3)
    return pts_arr, np.array(offs, dtype=np.float64)


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def synchronize(imus, scans, max_frames=None):
    """Port of MeasurementSynchronizer::synchronizeIMULiDAR.

    Returns (scan, imu_block) frames; IMU is partitioned non-overlapping across scans.
    The batch pass is just the online synchronizer fed everything at once, so the
    two stay bit-for-bit equivalent (single source of the sync rule).
    """
    from se3_lio.online_sync import OnlineSynchronizer

    sync = OnlineSynchronizer()
    for row in imus:
        sync.add_imu(row)
    for scan in scans:
        sync.add_scan(scan["header_ts"], scan["pts"], scan["offsets"])
    frames = sync.drain()
    if max_frames:
        frames = frames[:max_frames]
    return frames


def stream_frames(bag_path, imu_topic, lidar_topic, min_range, max_frames=None, point_filter_num=1):
    """Yield synced `Frame`s one at a time in bounded memory: the bag is read one
    message at a time (never buffered whole), and the online synchronizer drains
    each scan as soon as it is IMU-covered -- so processed scans are dropped
    instead of accumulating and large bags no longer OOM. The emitted frames are
    identical to the batch `synchronize` path (both apply the same rule, pinned by
    tests/test_online_sync.py)."""
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu
    from livox_ros_driver2.msg import CustomMsg
    from se3_lio.online_sync import OnlineSynchronizer

    sync = OnlineSynchronizer()
    emitted = 0
    reader = _open_reader(bag_path)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == imu_topic:
            m = deserialize_message(data, Imu)
            a, w = m.linear_acceleration, m.angular_velocity
            sync.add_imu([_stamp(m.header), a.x, a.y, a.z, w.x, w.y, w.z])
        elif topic == lidar_topic:
            m = deserialize_message(data, CustomMsg)
            pts, offs = _convert_livox(m, min_range, point_filter_num)
            sync.add_scan(_stamp(m.header), pts, offs)
        else:
            continue
        # Drain after every message: a buffered scan is emitted as soon as an IMU
        # covers its end (so tail scans need no separate final flush).
        for scan, imu_block in sync.drain():
            yield Frame(points=scan["pts"], point_times=scan["offsets"],
                        imu=imu_block, stamp=scan["header_ts"])
            emitted += 1
            if max_frames and emitted >= max_frames:
                return


class RosbagDataset:
    """Streaming iterable of synced SE(3)-LIO `Frame`s from a ROS2 rosbag. The bag
    is read once per iteration, one frame at a time -- no full-bag materialization,
    so RAM stays bounded regardless of bag size."""

    def __init__(self, bag_path, imu_topic, lidar_topic, min_range, max_frames=None, point_filter_num=1):
        self._args = (bag_path, imu_topic, lidar_topic, min_range, max_frames, point_filter_num)

    def __iter__(self):
        return stream_frames(*self._args)
