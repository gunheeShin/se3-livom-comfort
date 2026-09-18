"""ROS1 rosbag dataset (Ouster PointCloud2) for SE(3)-LIO.

Reproduces the ROS1 node's data path:
  - ros1_conversion.h::convertIMUMessage / convertOusterMessage
  - MeasurementSynchronizer::synchronizeIMULiDAR (shared core, ported in rosbag.py)

ROS1 `.bag` files are read with the `rosbags` library (pure Python, no ROS1
install needed). Ouster clouds are `sensor_msgs/PointCloud2` with per-point `t`
(uint32, ns offset from scan start).
"""

from pathlib import Path

import numpy as np

from se3_lio.datasets.rosbag import Frame
from se3_lio.online_sync import OnlineSynchronizer

# sensor_msgs/PointField datatype -> numpy little-endian format
_PF_NP = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def _stamp(header):
    """Header stamp in seconds (ROS2 ``nanosec`` or ROS1 ``nsec``)."""
    s = header.stamp
    ns = getattr(s, "nanosec", None)
    if ns is None:
        ns = getattr(s, "nsec", 0)
    return s.sec + ns * 1e-9


def _pc2_arrays(msg, needed):
    """Structured view of a PointCloud2 over `needed` fields (zero-copy)."""
    fmt = {f.name: (f.offset, _PF_NP[f.datatype]) for f in msg.fields if f.name in needed}
    dt = np.dtype(
        {
            "names": list(fmt),
            "formats": [fmt[n][1] for n in fmt],
            "offsets": [fmt[n][0] for n in fmt],
            "itemsize": msg.point_step,
        }
    )
    return np.frombuffer(bytes(msg.data), dtype=dt)


def _xyz_keep(arr, min_range, point_filter_num=1):
    xyz = np.stack(
        [arr["x"].astype(np.float64), arr["y"].astype(np.float64), arr["z"].astype(np.float64)],
        axis=1,
    )
    # Range test in float32 to match the ROS node bit-for-bit: CustomPointType is
    # float, so the C++ converters compute x*x+y*y+z*z in float32. Doing it in
    # float64 here flips 1-2 boundary points per scan, which a near-unstable run
    # amplifies into trajectory divergence. Cast x/y/z (already float32) and keep
    # the float32 sum's order, then promote for the compare (NaN-safe). [[tartandrive-dataset-setup]]
    xf, yf, zf = arr["x"], arr["y"], arr["z"]
    d2 = (xf * xf + yf * yf + zf * zf).astype(np.float64)
    keep = d2 > (min_range * min_range)
    if point_filter_num > 1:  # keep every Nth raw point (i += N in the ROS node)
        keep &= (np.arange(len(keep)) % point_filter_num) == 0
    return xyz, keep


def _convert_ouster(msg, min_range, point_filter_num=1):
    """Port of convertOusterMessage: per-point time is the `t` field (ns offset
    from scan start). Original order preserved (so the last kept point matches
    C++ `points.back()` used by the synchronizer)."""
    arr = _pc2_arrays(msg, ("x", "y", "z", "intensity", "t"))
    xyz, keep = _xyz_keep(arr, min_range, point_filter_num)
    offs = arr["t"].astype(np.float64) * 1e-9
    return xyz[keep], offs[keep], 0.0


def _convert_hesai(msg, min_range, point_filter_num=1):
    """Port of convertHesaiMessage: per-point `timestamp` is absolute seconds,
    so the offset from scan start is timestamp - header stamp (GrandTour Hesai)."""
    arr = _pc2_arrays(msg, ("x", "y", "z", "intensity", "timestamp"))
    xyz, keep = _xyz_keep(arr, min_range, point_filter_num)
    offs = arr["timestamp"].astype(np.float64) - _stamp(msg.header)
    return xyz[keep], offs[keep], 0.0


def _convert_velodyne(msg, min_range, point_filter_num=1):
    """Port of convertVelodyneMessage: per-point `time` (float32 seconds) is
    relative to the header stamp (scan end), in [-period, 0]. Shift by the
    minimum time so offsets are ascending and >= 0 and the header marks the scan
    start; t0 is taken over all points (before filtering) for binding parity."""
    arr = _pc2_arrays(msg, ("x", "y", "z", "intensity", "time"))
    xyz, keep = _xyz_keep(arr, min_range, point_filter_num)
    t = arr["time"].astype(np.float64)
    header_shift = float(t.min()) if t.size else 0.0
    offs = t - header_shift
    return xyz[keep], offs[keep], header_shift


def _convert_lidar(msg, min_range, point_filter_num=1):
    """Dispatch by per-point time field, returning (xyz, offsets, header_shift):
    Ouster has `t` (uint32 ns), Hesai/GrandTour has `timestamp` (float64 absolute
    seconds), Velodyne has `time` (float32 seconds relative to scan end). The
    header shift is added to the scan stamp so it marks the scan start."""
    names = {f.name for f in msg.fields}
    if "t" in names:
        return _convert_ouster(msg, min_range, point_filter_num)
    if "timestamp" in names:
        return _convert_hesai(msg, min_range, point_filter_num)
    if "time" in names:
        return _convert_velodyne(msg, min_range, point_filter_num)
    raise RuntimeError(f"no per-point time field (t/timestamp/time) in cloud: {sorted(names)}")


def _as_bag_list(bag):
    """Resolve a bag spec to a list of Paths. Accepts a single path, a list of
    paths, or a glob string. Multiple bags (e.g. LiDAR + IMU split across files,
    or time chunks) are merged in log-time order by AnyReader."""
    import glob

    if not isinstance(bag, (str, Path)):
        return [Path(b) for b in bag]
    s = str(bag)
    if any(c in s for c in "*?["):
        matches = sorted(glob.glob(s))
        if not matches:
            raise FileNotFoundError(f"no bags match glob: {s}")
        return [Path(m) for m in matches]
    return [Path(s)]


def stream_frames(bag_path, imu_topic, lidar_topic, min_range, max_frames=None, point_filter_num=1):
    """Yield synced `Frame`s one at a time in bounded memory: the bag is read one
    message at a time (never buffered whole), and the online synchronizer drains
    each scan as soon as it is IMU-covered -- so processed scans are dropped
    instead of accumulating and large bags no longer OOM. The emitted frames are
    identical to the batch `synchronize` path (both apply the same rule, pinned by
    tests/test_online_sync.py)."""
    from rosbags.highlevel import AnyReader

    sync = OnlineSynchronizer()
    emitted = 0
    with AnyReader(_as_bag_list(bag_path)) as reader:
        conns = [c for c in reader.connections if c.topic in (imu_topic, lidar_topic)]
        if not conns:
            raise RuntimeError(f"topics not found in bag: imu={imu_topic} lidar={lidar_topic}")
        for conn, _t, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            if conn.topic == imu_topic:
                a, w = m.linear_acceleration, m.angular_velocity
                sync.add_imu([_stamp(m.header), a.x, a.y, a.z, w.x, w.y, w.z])
            else:
                pts, offs, shift = _convert_lidar(m, min_range, point_filter_num)
                sync.add_scan(_stamp(m.header) + shift, pts, offs)
            # Drain after every message: a buffered scan is emitted as soon as an
            # IMU covers its end (so tail scans need no separate final flush).
            for scan, imu_block in sync.drain():
                yield Frame(points=scan["pts"], point_times=scan["offsets"],
                            imu=imu_block, stamp=scan["header_ts"])
                emitted += 1
                if max_frames and emitted >= max_frames:
                    return


class Ros1BagDataset:
    """Streaming iterable of synced SE(3)-LIO `Frame`s from a ROS1 (.bag) sequence.
    The bag is read once per iteration, one frame at a time -- no full-bag
    materialization, so RAM stays bounded regardless of bag size."""

    def __init__(self, bag_path, imu_topic, lidar_topic, min_range, max_frames=None, point_filter_num=1):
        self._args = (bag_path, imu_topic, lidar_topic, min_range, max_frames, point_filter_num)

    def __iter__(self):
        return stream_frames(*self._args)
