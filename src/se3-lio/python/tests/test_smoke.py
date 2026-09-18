import numpy as np

from se3_lio import SE3LIO, SE3LIOConfig
from se3_lio import se3_lio_pybind as lio
from se3_lio.pipeline import _rot_to_quat_xyzw
from se3_lio.config import _quat_wxyz_to_rot


def _dummy_frame():
    rng = np.random.default_rng(0)
    n = 2000
    points = rng.uniform(-20.0, 20.0, size=(n, 3))
    point_times = np.linspace(0.0, 0.1, n)
    m = 20
    imu = np.zeros((m, 7))
    imu[:, 0] = np.linspace(0.0, 0.1, m)
    imu[:, 3] = 9.81
    return points, point_times, imu


def test_highlevel_wrapper():
    config = SE3LIOConfig(max_iter=4, downsample_resolution=0.5)
    odom = SE3LIO(config, np.eye(4))
    points, point_times, imu = _dummy_frame()
    state, cloud = odom.register_frame(points, point_times, imu, frame_stamp=0.0)
    assert state.pose.shape == (4, 4)
    assert cloud.ndim == 2 and cloud.shape[1] == 3  # deskewed body-frame cloud


def test_config_to_pybind():
    config = SE3LIOConfig(acc_noise=0.2, voxel_map_layer_size=[4, 4])
    c = config.to_pybind()
    assert c.acc_noise == 0.2
    assert list(c.voxel_map_layer_size) == [4, 4]


def test_register_frame_returns_state():
    config = lio._SE3LIOConfig()
    config.max_iter = 4
    config.downsample_resolution = 0.5

    odom = lio._SE3LIO(config, np.eye(4))

    rng = np.random.default_rng(0)
    n = 2000
    points = rng.uniform(-20.0, 20.0, size=(n, 3))
    point_times = np.linspace(0.0, 0.1, n)

    # gravity-aligned, otherwise static IMU spanning the frame
    m = 20
    t = np.linspace(0.0, 0.1, m)
    imu = np.zeros((m, 7))
    imu[:, 0] = t
    imu[:, 3] = 9.81  # az

    state, cloud = odom._register_frame(points, point_times, imu, frame_stamp=0.0)

    assert state.pose.shape == (4, 4)
    assert np.allclose(state.pose[3], [0.0, 0.0, 0.0, 1.0])
    assert state.covariance.shape == (17, 17)  # SE3(6)+vel(3)+bg(3)+ba(3)+S2 grav(2)
    assert cloud.ndim == 2 and cloud.shape[1] == 3  # deskewed body-frame cloud


def test_config_roundtrip():
    config = lio._SE3LIOConfig()
    config.voxel_map_layer_size = [4, 4, 3]
    assert list(config.voxel_map_layer_size) == [4, 4, 3]


def test_rot_to_quat_roundtrip():
    # 90 deg about z: rot -> quat(xyzw) -> rot round-trips (used by save_tum + the nodes).
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    x, y, z, w = _rot_to_quat_xyzw(R)
    assert np.allclose(_quat_wxyz_to_rot([w, x, y, z]), R, atol=1e-9)
