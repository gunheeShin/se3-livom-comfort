"""Typed configuration for SE(3)-LIO and a loader for the node's params.yaml."""

from typing import List

import numpy as np
from pydantic import BaseModel, Field

from se3_lio.pybind import se3_lio_pybind as _pb


class VisualConfig(BaseModel):
    """Mirrors VisualUpdateConfig (cpp/se3_lio/visual/visual_update.h)."""

    patch_size: int = 8
    pyramid_levels: int = 3
    grid_size: int = 40
    max_iter: int = 5
    img_point_cov: float = 100.0
    outlier_threshold: float = 1000.0
    depth_continuous_thresh: float = 0.5
    voxel_size: float = 0.5
    max_obs: int = 30
    max_view_angle_deg: float = 60.0
    max_gyro_deg: float = 0.0
    keep_cov: bool = False
    probe: bool = False
    pose_only: bool = False
    verbose: bool = False

    def to_pybind(self) -> "_pb._VisualConfig":
        c = _pb._VisualConfig()
        for field in type(self).model_fields:
            setattr(c, field, getattr(self, field))
        return c


class CameraConfig(BaseModel):
    """Mirrors CameraConfig (SE3_LIO.h): pinhole K after undistortion, camera <- body extrinsic."""

    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    dist: List[float] = Field(default_factory=list)
    fisheye: bool = False
    T_cam_imu: List[List[float]] = Field(default_factory=lambda: np.eye(4).tolist())

    def to_pybind(self) -> "_pb._CameraConfig":
        c = _pb._CameraConfig()
        for field in type(self).model_fields:
            if field != "T_cam_imu":
                setattr(c, field, getattr(self, field))
        c.T_cam_imu = np.asarray(self.T_cam_imu, dtype=float).reshape(4, 4)
        return c


class SE3LIOConfig(BaseModel):
    """Mirrors SE3_LIO_Config (cpp/se3_lio/pipeline/SE3_LIO.h) with defaults."""

    acc_noise: float = 0.1
    gyr_noise: float = 0.1
    bg_noise: float = 0.0001
    ba_noise: float = 0.0001

    lidar_range_noise: float = 0.001
    lidar_angle_noise: float = 0.01
    lidar_range_noises: List[float] = Field(default_factory=list)  # per-LiDAR, indexed by Frame.lidar_idx
    lidar_angle_noises: List[float] = Field(default_factory=list)
    lidar_extrinsics: List[List[List[float]]] = Field(default_factory=list)  # per-LiDAR 4x4 sensor->body (register_multi)
    lidar_min_ranges: List[float] = Field(default_factory=list)  # per-LiDAR near cut [m] in mergeLiDARs

    downsample_resolution: float = 0.5
    downsample_centroid: bool = False
    max_iter: int = 4

    voxel_map_resolution: float = 1.0
    voxel_map_max_layer: int = 2
    voxel_map_layer_size: List[int] = Field(default_factory=lambda: [5, 5, 5, 5, 5])
    voxel_map_max_point_size: int = 1000
    voxel_map_plane_thres: float = 0.01

    voxel_map_sliding_en: bool = False
    voxel_map_sliding_thresh: float = 8.0
    voxel_map_half_size: int = 50

    verbose: bool = False

    residual_floor: float = 0.0   # FL2-style A/B switches, see SE3_LIO_Config
    deskew_cov_en: bool = True
    init_fl2: bool = False
    const_r: bool = False

    visual_en: bool = False  # photometric update; cameras are filled by the runner from calib.json
    visual: VisualConfig = Field(default_factory=VisualConfig)
    cameras: List[CameraConfig] = Field(default_factory=list)

    def to_pybind(self) -> "_pb._SE3LIOConfig":
        c = _pb._SE3LIOConfig()
        for field in type(self).model_fields:
            if field not in ("visual", "cameras"):
                setattr(c, field, getattr(self, field))
        c.lidar_extrinsics = [np.asarray(T, dtype=float).reshape(4, 4) for T in self.lidar_extrinsics]
        c.visual = self.visual.to_pybind()
        c.cameras = [cam.to_pybind() for cam in self.cameras]
        return c


def _quat_wxyz_to_rot(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _get(d, dotted, default):
    cur = d
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _scalar(v):
    return float(v[0]) if isinstance(v, (list, tuple)) else float(v)


def _vector(v):
    return [float(x) for x in v] if isinstance(v, (list, tuple)) else []


def load_node_params(params_path):
    """Parse the ROS2 node's params.yaml the same way lio_node.cpp does.

    Returns dict: config (SE3LIOConfig), extrinsic (4x4), min_range, imu_topic,
    lidar_topic. Reproduces the node's exact key mapping, including
    b_acc_cov -> bg_noise / b_gyr_cov -> ba_noise and the range_cov/angle_cov
    fallback defaults (the yaml's ranging_covs/angle_covs are not read).
    """
    import yaml

    with open(params_path) as f:
        raw = yaml.safe_load(f)
    # ROS2 params.yaml nests under "/**: ros__parameters:"; ROS1 yaml is flat.
    node = raw.get("/**", raw)
    params = node.get("ros__parameters", node)

    config = SE3LIOConfig(
        acc_noise=_get(params, "sensors.imu.acc_cov", 0.1),
        gyr_noise=_get(params, "sensors.imu.gyr_cov", 0.1),
        bg_noise=_get(params, "sensors.imu.b_acc_cov", 0.0001),
        ba_noise=_get(params, "sensors.imu.b_gyr_cov", 0.0001),
        lidar_range_noise=_scalar(_get(params, "sensors.lidar.range_cov", 0.001)),
        lidar_angle_noise=_scalar(_get(params, "sensors.lidar.angle_cov", 0.01)),
        lidar_min_ranges=_vector(_get(params, "sensors.lidar.min_range", 0.1)),
        lidar_range_noises=_vector(_get(params, "sensors.lidar.range_cov", 0.001)),
        lidar_angle_noises=_vector(_get(params, "sensors.lidar.angle_cov", 0.01)),
        downsample_resolution=_get(params, "downsample.resolution", 0.5),
        downsample_centroid=bool(_get(params, "downsample.centroid", False)),
        max_iter=int(_get(params, "max_iter", 4)),
        voxel_map_resolution=_get(params, "voxel_map.resolution", 1.0),
        voxel_map_max_layer=int(_get(params, "voxel_map.max_layer", 2)),
        voxel_map_layer_size=[int(x) for x in _get(params, "voxel_map.layer_size", [5, 5, 5, 5, 5])],
        voxel_map_max_point_size=int(_get(params, "voxel_map.max_point_size", 1000)),
        voxel_map_plane_thres=float(_get(params, "voxel_map.plane_threshold", 0.01)),
        voxel_map_sliding_en=bool(_get(params, "voxel_map.map_sliding_en", False)),
        voxel_map_sliding_thresh=float(_get(params, "voxel_map.sliding_thresh", 8.0)),
        voxel_map_half_size=int(_get(params, "voxel_map.half_map_size", 50)),
        verbose=bool(_get(params, "verbose", False)),
        residual_floor=float(_get(params, "residual_floor", 0.0)),
        deskew_cov_en=bool(_get(params, "deskew_cov_en", True)),
        init_fl2=bool(_get(params, "init_fl2", False)),
        const_r=bool(_get(params, "const_r", False)),
        visual_en=bool(_get(params, "visual.enable", False)),
        visual=VisualConfig(**{k: v for k, v in (_get(params, "visual", {}) or {}).items() if k != "enable"}),
    )

    t = _get(params, "sensors.t_exts", [0.0, 0.0, 0.0])
    q = _get(params, "sensors.q_exts", [1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = t
    extrinsic[:3, :3] = _quat_wxyz_to_rot(q)

    return {
        "config": config,
        "extrinsic": extrinsic,
        "min_range": _scalar(_get(params, "sensors.lidar.min_range", 0.1)),
        "point_filter_num": int(_get(params, "sensors.lidar.point_filter_num", 1)),
        "imu_topic": _get(params, "imu_topic", "/imu/data"),
        "lidar_topic": _get(params, "lidar_topic", "/lidar/points"),
    }
