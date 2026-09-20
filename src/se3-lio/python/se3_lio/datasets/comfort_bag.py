"""GrandTour mission bags played at sensor time (online-bag) as SE(3)-LIO frames — no extraction step.

<mission>/*_hesai.bag        /boxi/hesai/points                             PointCloud2, XT32 layout
          *_livox.bag        /boxi/livox/points                             PointCloud2, timestamp in ns
          *_stim320_imu.bag  /boxi/stim320/imu
          *_alphasense.bag   /boxi/alphasense/<cam>/image_raw/compressed    (+ camera_info)
          *_tf_minimal.bag   /tf_static

One thread per bag releases each message at the wall-clock moment it would exist on the robot (a scan once
its last point is measured, an image or IMU sample at its stamp) into a bounded queue. A full queue blocks
the reader: nothing is dropped, so a pipeline that cannot keep up shows as growing latency, never as different
frames. The frames come from comfort_offline.epoch_frames — the rule the extracted files go through — and the
arrays are built exactly as tools/extract.py + comfort_offline._read_bin build them, so both modes give the
same trajectory bit for bit.
"""

import glob
import itertools
import os
import queue
import threading
import time

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from se3_lio.datasets.comfort_offline import LIVOX_RING, XT32, _ImageLoader, _Points, _filter, epoch_frames

HESAI, LIVOX, IMU = "/boxi/hesai/points", "/boxi/livox/points", "/boxi/stim320/imu"
IMU_FRAME, LIDAR_FRAME, LIVOX_FRAME, PRISM_FRAME = "stim320_imu", "hesai_lidar", "livox_lidar", "prism"
_PF = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}
_QUEUE = {"hesai": 20, "livox": 20, "imu": 2000, "image": 20}  # a few seconds per sensor


def _bag(mdir, suffix):
    m = glob.glob(os.path.join(mdir, f"*_{suffix}.bag"))
    if not m:
        raise FileNotFoundError(f"{suffix}.bag not found in {mdir}")
    return m[0]


def _messages(path, topics):
    with Reader(path) as r:
        ts = get_typestore(Stores.ROS1_NOETIC)
        add = {}
        for c in r.connections:
            d = getattr(c.msgdef, "data", c.msgdef)
            if d:
                add.update(get_types_from_msg(d, c.msgtype))
        ts.register(add)
        for c, _, raw in r.messages(connections=[c for c in r.connections if c.topic in topics]):
            yield c.topic, ts.deserialize_ros1(raw, c.msgtype)


def _ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _cloud(m):
    src = np.dtype({"names": [f.name for f in m.fields], "formats": [_PF[f.datatype] for f in m.fields],
                    "offsets": [f.offset for f in m.fields], "itemsize": m.point_step})
    return np.frombuffer(m.data, dtype=src, count=m.width * m.height)


def _hesai_xt32(m):
    pts = _cloud(m)
    o = np.empty(len(pts), dtype=XT32)
    for k in XT32.names:
        o[k] = pts[k]
    return o


def _livox_xt32(m):
    pts = _cloud(m)
    pts = pts[np.isfinite(pts["x"]) & np.isfinite(pts["y"]) & np.isfinite(pts["z"])]
    o = np.empty(len(pts), dtype=XT32)
    o["x"], o["y"], o["z"], o["intensity"] = pts["x"], pts["y"], pts["z"], pts["intensity"]
    o["ring"] = LIVOX_RING
    o["timestamp"] = pts["timestamp"].astype(np.float64) * 1e-9
    return o[np.argsort(o["timestamp"], kind="stable")]


def _quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _chain(edges, src, dst):
    """T_src_dst over the /tf_static tree (BFS, edges usable both ways)."""
    adj = {}
    for (a, b), T in edges.items():
        adj.setdefault(a, []).append((b, T))
        adj.setdefault(b, []).append((a, np.linalg.inv(T)))
    seen, q = {src: np.eye(4)}, [src]
    while q:
        a = q.pop(0)
        if a == dst:
            return seen[a]
        for b, T in adj.get(a, []):
            if b not in seen:
                seen[b] = seen[a] @ T
                q.append(b)
    raise RuntimeError(f"no tf path {src} -> {dst}")


def read_calib(mdir, cam=None):
    """The calib.json tools/extract.py writes, straight from the bags: T_imu_lidar, T_imu_prism,
    T_lidar_livox and, with `cam`, cams[cam] = camera_info + T_cam_lidar."""
    edges = {}
    for _, m in _messages(_bag(mdir, "tf_minimal"), ["/tf_static"]):
        for tr in m.transforms:
            p, q = tr.transform.translation, tr.transform.rotation
            T = np.eye(4)
            T[:3, :3] = _quat_to_R(q.x, q.y, q.z, q.w)
            T[:3, 3] = [p.x, p.y, p.z]
            edges.setdefault((tr.header.frame_id, tr.child_frame_id), T)
    calib = {"T_imu_lidar": _chain(edges, IMU_FRAME, LIDAR_FRAME).tolist(),
             "T_imu_prism": _chain(edges, IMU_FRAME, PRISM_FRAME).tolist(),
             "T_lidar_livox": _chain(edges, LIDAR_FRAME, LIVOX_FRAME).tolist(), "cams": {}}
    if cam:
        _, m = next(_messages(_bag(mdir, "alphasense"), [f"/boxi/alphasense/{cam}/camera_info"]))
        info = {"frame": m.header.frame_id, "model": m.distortion_model, "width": int(m.width), "height": int(m.height),
                "K": [float(v) for v in m.K], "D": [float(v) for v in m.D]}
        info["T_cam_lidar"] = _chain(edges, info["frame"], LIDAR_FRAME).tolist()
        calib["cams"][cam] = info
    return calib


class ComfortBagDataset:
    """Hesai (+ Livox) + STIM320 + one camera from the mission bags, played at 1x. Iterate once."""

    def __init__(self, mission_dir, min_range, imu_dt=0.0, max_frames=None, point_filter_num=1, livox=True,
                 cam="front_center", img_time_offset=0.0, cam_calib=None):
        self._mdir, self._min_range, self._imu_dt, self._max_frames = mission_dir, min_range, imu_dt, max_frames
        self._pfn, self._cam, self._img_dt = point_filter_num, cam, img_time_offset
        self._loaders = [_ImageLoader(cam_calib)] if cam_calib else None
        self._names = ["hesai", "imu", "image"] + (["livox"] if livox else [])
        self._q = {n: queue.Queue(_QUEUE[n]) for n in self._names}
        suffix = {"hesai": "hesai", "livox": "livox", "imu": "stim320_imu", "image": "alphasense"}
        with_times = []
        for n in self._names:
            with Reader(_bag(mission_dir, suffix[n])) as r:
                with_times.append(r.start_time)
        self._data0 = min(with_times) * 1e-9  # data time that maps to wall0
        self._wall0 = None

    def lag(self, stamp, wall):
        """Seconds between the instant `stamp` happened (on the played clock) and wall time `wall`."""
        return wall - (self._wall0 + (stamp - self._data0))

    # --- reader side: (time the message exists, item) per sensor ---
    def _hesai(self):
        for _, m in _messages(_bag(self._mdir, "hesai"), [HESAI]):
            a = _hesai_xt32(m)
            if not len(a):
                continue
            yield float(a["timestamp"].max()), (float(a["timestamp"][0]), _filter(a, self._min_range, self._pfn))

    def _livox(self):
        for _, m in _messages(_bag(self._mdir, "livox"), [LIVOX]):
            a = _livox_xt32(m)
            if len(a):
                yield float(a["timestamp"][-1]), _filter(a, self._min_range, 1)

    def _imu(self):
        for _, m in _messages(_bag(self._mdir, "stim320_imu"), [IMU]):
            w, a = m.angular_velocity, m.linear_acceleration
            t = float(_ns(m.header)) * 1e-9
            yield t, np.array([t + self._imu_dt, a.x, a.y, a.z, w.x, w.y, w.z])

    def _image(self):
        for _, m in _messages(_bag(self._mdir, "alphasense"), [f"/boxi/alphasense/{self._cam}/image_raw/compressed"]):
            t = float(_ns(m.header)) * 1e-9
            yield t, (t + self._img_dt, [m.data.tobytes()])

    def _play(self, name):
        q = self._q[name]
        try:
            for t, item in getattr(self, f"_{name}")():
                wait = self._wall0 + (t - self._data0) - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                q.put(item)
            q.put(None)
        except BaseException as e:  # surfaces on the pipeline's thread
            q.put(e)

    # --- pipeline side ---
    def _arrivals(self, name):
        while True:
            item = self._q[name].get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def __iter__(self):
        self._wall0 = time.perf_counter() + 1.0
        for n in self._names:
            threading.Thread(target=self._play, args=(n,), name=f"bag-{n}", daemon=True).start()
        hesai = self._arrivals("hesai")
        first = next(hesai)  # the first raw Hesai point opens the run, as in comfort_offline._boundaries
        scans = itertools.chain([first[1]], (s for _, s in hesai))
        aux = [_Points(self._arrivals("livox"))] if "livox" in self._names else []
        return epoch_frames(first[0], self._arrivals("image"), self._arrivals("imu"), _Points(scans), aux,
                            self._loaders, True, self._max_frames)
