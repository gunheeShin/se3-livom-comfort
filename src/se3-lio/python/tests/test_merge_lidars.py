import numpy as np

from se3_lio import SE3LIO, SE3LIOConfig


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_merge_matches_reference():
    rng = np.random.default_rng(1)
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, :3] = _rot_z(30)
    T1[:3, 3] = [0.1, -0.2, 0.3]
    config = SE3LIOConfig(lidar_extrinsics=[T0.tolist(), T1.tolist()])
    odom = SE3LIO(config)

    p0 = rng.uniform(-10, 10, (500, 3)); t0 = np.sort(rng.uniform(0, 0.1, 500)); s0 = 100.0
    p1 = rng.uniform(-10, 10, (300, 3)); t1 = np.sort(rng.uniform(0, 0.1, 300)); s1 = 100.03
    merged, stamp = odom.merge_lidars([(p0, t0, s0), (p1, t1, s1)])

    # reference: apply extrinsic, tag idx, re-base offsets to earliest start, stable sort by time
    ref = np.concatenate([
        np.column_stack([p0 @ T0[:3, :3].T + T0[:3, 3], np.zeros(500), t0 + (s0 - s0)]),
        np.column_stack([p1 @ T1[:3, :3].T + T1[:3, 3], np.ones(300), t1 + (s1 - s0)]),
    ])
    ref = ref[np.argsort(ref[:, 4], kind="stable")]

    assert stamp == s0
    assert merged.shape == ref.shape
    assert merged.shape[0] == 800  # no min_range configured -> nothing cut
    np.testing.assert_allclose(merged[:, :3], ref[:, :3], atol=1e-4)  # float32 storage in the core
    np.testing.assert_array_equal(merged[:, 3], ref[:, 3])
    np.testing.assert_allclose(merged[:, 4], ref[:, 4], atol=1e-9)


def test_merge_min_range_cut():
    T = np.eye(4).tolist()
    odom = SE3LIO(SE3LIOConfig(lidar_extrinsics=[T, T], lidar_min_ranges=[0.0, 1.0]))
    p0 = np.array([[0.1, 0, 0], [5, 0, 0]]); p1 = np.array([[0.5, 0, 0], [5, 0, 0], [0, 0.99, 0]])
    merged, _ = odom.merge_lidars([(p0, np.zeros(2), 0.0), (p1, np.zeros(3), 0.0)])
    assert merged.shape[0] == 3  # lidar 1 keeps only the 5 m point
    np.testing.assert_array_equal(merged[:, 3], [0, 0, 1])


def test_register_multi_runs():
    config = SE3LIOConfig(lidar_extrinsics=[np.eye(4).tolist(), np.eye(4).tolist()])
    odom = SE3LIO(config)
    rng = np.random.default_rng(0)
    imu = np.zeros((20, 7)); imu[:, 0] = np.linspace(0.0, 0.1, 20); imu[:, 3] = 9.81
    scans = [(rng.uniform(-20, 20, (1000, 3)), np.linspace(0, 0.1, 1000), 0.0),
             (rng.uniform(-20, 20, (600, 3)), np.linspace(0, 0.1, 600), 0.01)]
    state, cloud = odom.register_multi(scans, imu)
    assert state.pose.shape == (4, 4)
    assert cloud.shape[1] == 3
