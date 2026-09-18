"""comfort_ws offline extraction (tools/extract.py output) as SE(3)-LIO frames.

<mission>/comfort_offline/
  imu.txt           t_ns wx wy wz ax ay az            (STIM320)
  lidar*/<t_ns>.bin XT32 26B: x y z f32, intensity f32, ring u16, timestamp f64 (absolute s)
  livox/<t_ns>.bin  same layout, Livox frame, ring 200
  cam/<t_ns>.jpg    front_center images; cam_<name>/ other cameras (same trigger -> same stamps)

Epochs: without cam_dir each Hesai file is one epoch (state at the last point). With cam_dir
the image stamps are the epoch boundaries: epoch k holds every point in [t_img_{k-1}, t_img_k)
and the state is propagated to exactly t_img_k (Frame.end_time), as FAST-LIVO2 does. The first
cam_dir sets the epochs; with cam_calibs the images are decoded, undistorted onto the same K
and attached as Frame.grays (one per cam_dir, same order) for the photometric update.
Merged scans (tools/merge_livox.py) tag Livox points with ring 200 -> lidar_idx 1.
"""

import glob
import os

import numpy as np

from se3_lio.datasets.rosbag import Frame
from se3_lio.online_sync import OnlineSynchronizer

LIVOX_RING = 200

XT32 = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "ring", "timestamp"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f8"],
        "offsets": [0, 4, 8, 12, 16, 18],
        "itemsize": 26,
    }
)


def _files(d, ext="bin"):
    return sorted(glob.glob(os.path.join(d, f"*.{ext}")), key=lambda f: int(os.path.basename(f).split(".")[0]))


def _read_imu(path, imu_dt):
    a = np.loadtxt(path, dtype=np.float64)
    rows = np.empty((len(a), 7))
    rows[:, 0] = a[:, 0] * 1e-9 + imu_dt
    rows[:, 1:4] = a[:, 4:7]  # acc
    rows[:, 4:7] = a[:, 1:4]  # gyr
    return rows


def _read_bin(path, min_range, point_filter_num):
    a = np.fromfile(path, dtype=XT32)
    xf, yf, zf = a["x"], a["y"], a["z"]
    keep = (xf * xf + yf * yf + zf * zf).astype(np.float64) > min_range * min_range
    if point_filter_num > 1:
        keep &= (np.arange(len(keep)) % point_filter_num) == 0
    a = a[keep]
    xyz = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    return a["timestamp"].astype(np.float64), xyz, a["ring"]


class _Points:
    """Time-ordered points of one sensor directory, consumed up to a boundary."""

    def __init__(self, d, min_range, point_filter_num=1):
        self._args = (min_range, point_filter_num)
        self._files = iter(_files(d))
        self._t, self._xyz, self._ring = np.empty(0), np.empty((0, 3)), np.empty(0, np.uint16)
        self._done = False

    def take(self, t_end):
        """All points with t < t_end (in file order; files are time-monotonic and disjoint)."""
        while not self._done and (self._t.size == 0 or self._t[-1] < t_end):
            f = next(self._files, None)
            if f is None:
                self._done = True
                break
            t, xyz, ring = _read_bin(f, *self._args)
            self._t, self._xyz, self._ring = (
                np.concatenate([self._t, t]), np.concatenate([self._xyz, xyz]), np.concatenate([self._ring, ring]))
        m = self._t < t_end
        out = self._t[m], self._xyz[m], self._ring[m]
        self._t, self._xyz, self._ring = self._t[~m], self._xyz[~m], self._ring[~m]
        return out


def _boundaries(lidar_dir, cam_dirs, img_time_offset):
    """(epoch start, epoch ends, image paths per camera or None). Without cams the ends are the
    Hesai file starts. Cameras share the trigger, so the first directory's stamps set the epochs
    and the others are matched to them (missing frame -> None)."""
    files = _files(lidar_dir)
    # first-point stamps as stored (the file name in ns rounds differently in float64)
    starts = np.array([np.fromfile(f, dtype=XT32, count=1)["timestamp"][0] for f in files])
    if cam_dirs:
        imgs = [_files(d, "*") for d in cam_dirs]
        ns = [np.array([int(os.path.basename(f).split(".")[0]) for f in fs]) for fs in imgs]
        stamps = ns[0] * 1e-9 + img_time_offset
        by_ns = [dict(zip(n, fs)) for n, fs in zip(ns, imgs)]
        paths = [[d.get(k) for d in by_ns] for k in ns[0]]
        return starts[0], stamps, paths
    last_end = np.fromfile(files[-1], dtype=XT32)["timestamp"][-1]
    return starts[0], np.append(starts[1:], last_end + 1e-3), None


class _ImageLoader:
    """jpg/png -> undistorted gray (same K; equidistant or radtan per calib "model")."""

    def __init__(self, cam):
        import cv2

        self._cv2 = cv2
        K = np.array(cam["K"], dtype=np.float64).reshape(3, 3)
        D = np.array(cam["D"], dtype=np.float64)
        size = (int(cam["width"]), int(cam["height"]))
        init = cv2.fisheye.initUndistortRectifyMap if cam["model"] == "equidistant" else cv2.initUndistortRectifyMap
        self._maps = init(K, D, np.eye(3), K, size, cv2.CV_32FC1)

    def __call__(self, path):
        cv2 = self._cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)


class MultiFrame:
    """Several sensor scans sharing one IMU block: scans = [(points, point_times, stamp), ...]."""

    def __init__(self, scans, imu, end_time=None, grays=None):
        self.scans = scans
        self.imu = imu
        self.stamp = scans[0][2]
        self.end_time = end_time
        self.grays = grays


def stream_frames(offline_dir, lidar_dir, min_range, imu_dt=0.0, max_frames=None, point_filter_num=1,
                  livox_dir=None, cam_dirs=None, img_time_offset=0.0, cam_calibs=None):
    """Yields Frame (single lidar_dir, merged .bin or Hesai only) or MultiFrame (livox_dir given:
    Hesai + Livox as two scans merged inside the core). cam_calibs (one per cam_dir) attach images."""
    sync = OnlineSynchronizer()
    for row in _read_imu(os.path.join(offline_dir, "imu.txt"), imu_dt):
        sync.add_imu(row)
    start, bounds, imgs = _boundaries(lidar_dir, cam_dirs, img_time_offset)
    exact_end = imgs is not None
    loaders = [_ImageLoader(c) for c in cam_calibs] if cam_calibs and exact_end else None
    hesai = _Points(lidar_dir, min_range, point_filter_num)
    livox = _Points(livox_dir, min_range) if livox_dir else None
    if livox is not None:
        livox.take(start)  # Livox before the first Hesai point is dropped
    pending, emitted = {}, 0
    for k, t_end in enumerate(bounds):
        ht, hxyz, hring = hesai.take(t_end)
        if ht.size == 0:
            if livox is not None:
                livox.take(t_end)
            continue
        h0 = float(ht[0])
        if livox is None:
            frame = Frame(points=hxyz, point_times=ht - h0, imu=None, stamp=h0,
                          lidar_idx=(hring == LIVOX_RING).astype(np.float64),
                          end_time=t_end if exact_end else None)
            last = float(ht[-1])
        else:
            lt, lxyz, _ = livox.take(t_end)
            scans = [(hxyz, ht - h0, h0)]
            last = float(ht[-1])
            if lt.size:
                scans.append((lxyz, lt - lt[0], float(lt[0])))
                last = max(last, float(lt[-1]))
            frame = MultiFrame(scans, None, end_time=t_end if exact_end else None)
        if loaders is not None:
            frame.grays = [ld(p) if p else None for ld, p in zip(loaders, imgs[k])]
        pending[h0] = frame
        sync.add_scan(h0, None, np.array([0.0, (t_end if exact_end else last) - h0]))
        for scan, imu_block in sync.drain():
            frame = pending.pop(scan["header_ts"])
            frame.imu = imu_block
            yield frame
            emitted += 1
            if max_frames and emitted >= max_frames:
                return


class ComfortOfflineDataset:
    def __init__(self, offline_dir, lidar_dir, min_range, imu_dt=0.0, max_frames=None, point_filter_num=1,
                 livox_dir=None, cam_dirs=None, img_time_offset=0.0, cam_calibs=None):
        self._kw = dict(offline_dir=offline_dir, lidar_dir=lidar_dir, min_range=min_range, imu_dt=imu_dt,
                        max_frames=max_frames, point_filter_num=point_filter_num, livox_dir=livox_dir,
                        cam_dirs=cam_dirs, img_time_offset=img_time_offset, cam_calibs=cam_calibs)
        self._n = len(_files(cam_dirs[0], "*")) if cam_dirs else len(_files(lidar_dir))

    def __len__(self):
        return self._n

    def __iter__(self):
        return stream_frames(**self._kw)
