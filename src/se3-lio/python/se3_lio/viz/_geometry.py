"""Geometry helpers shared by the viz backends (rerun_logger + polyscope_viz)."""

import numpy as np


def lidar_to_world(pose, extrinsic, pts):
    """Transform LiDAR-frame points (N, 3) into the world frame.

    Mirrors the binding's convention: extrinsic maps LiDAR->body, pose maps
    body->world, so world = pose @ extrinsic @ [pt; 1].
    """
    pose = np.asarray(pose, dtype=float)
    extrinsic = np.asarray(extrinsic, dtype=float)
    pts = np.asarray(pts, dtype=float)
    T = pose @ extrinsic
    return pts @ T[:3, :3].T + T[:3, 3]


def vec_align(a, b):
    """Rotation matrix R (3x3) with R @ a == b, for unit vectors a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        # Antiparallel (e.g. upside-down IMU): 180 deg about any axis ⊥ to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def gravity_align(grav):
    """Rotation that maps the world frame so gravity points along -z (z up)."""
    if grav is None:
        return np.eye(3)
    grav = np.asarray(grav, dtype=float)
    n = float(np.linalg.norm(grav))
    if n < 1e-9:
        return np.eye(3)
    return vec_align(grav / n, np.array([0.0, 0.0, -1.0]))
