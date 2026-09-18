"""SE(3)-aligned ATE evaluation for TUM trajectories.

Self-contained: numpy + stdlib only, no compiled binding, so this module can be
unit-tested on the host without the LIO core built.
"""

import numpy as np


def read_tum(path):
    """Return (stamps (N,), positions (N,3), quats_xyzw (N,4)).

    Enforces TUM format: at least 8 columns (t x y z qx qy qz qw) with the
    timestamp in SECONDS. A nanosecond timestamp (e.g. a raw NTU poses.txt) is
    rejected with a hint -- otherwise it silently fails association and the ATE
    comes back inf.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError(
            f"{path}: expected TUM 't x y z qx qy qz qw' (>=8 columns), got {data.shape[1]}"
        )
    if data[:, 0].max() > 1e14:
        raise ValueError(
            f"{path}: timestamps look like nanoseconds; convert to seconds (t/1e9) before use"
        )
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def associate(ta, tb, max_dt):
    """Nearest-timestamp association: list of (i, j) index pairs within max_dt."""
    pairs = []
    j = 0
    nb = len(tb)
    for i, t in enumerate(ta):
        while j + 1 < nb and abs(tb[j + 1] - t) <= abs(tb[j] - t):
            j += 1
        if abs(tb[j] - t) <= max_dt:
            pairs.append((i, j))
    return pairs


def quat_angle_deg(qa, qb):
    """Angle of relative rotation between two unit quaternions (xyzw), degrees."""
    dot = np.abs(np.sum(qa * qb, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def umeyama_rigid(src, dst):
    """Closed-form rigid alignment (scale fixed = 1) mapping src -> dst.

    src, dst: (N, 3). Returns (R (3,3), t (3,)) minimizing ||R @ src + t - dst||.
    Reflection is removed via a determinant correction on the SVD.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    cov = (dst - mu_d).T @ (src - mu_s) / src.shape[0]
    U, _, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    t = mu_d - R @ mu_s
    return R, t


def evaluate(est_tum, gt_tum, max_dt=0.02, min_assoc=10):
    """Rigid-aligned ATE of estimate vs ground truth.

    Returns a dict with ate_rmse/mean/median/max (m), rot_rmse_deg, and counts.
    If too few associations are found, ate_rmse is inf so callers treat the
    combo as a failure.
    """
    te, pe, qe = read_tum(est_tum)
    tg, pg, qg = read_tum(gt_tum)
    pairs = associate(te, tg, max_dt)

    result = {
        "ate_rmse": float("inf"),
        "ate_mean": float("inf"),
        "ate_median": float("inf"),
        "ate_max": float("inf"),
        "rot_rmse_deg": float("inf"),
        "n_assoc": len(pairs),
        "n_est": int(len(te)),
        "n_gt": int(len(tg)),
    }
    if len(pairs) < min_assoc:
        return result

    ie = [i for i, _ in pairs]
    ig = [j for _, j in pairs]
    src = pe[ie]
    dst = pg[ig]

    R, t = umeyama_rigid(src, dst)
    aligned = (R @ src.T).T + t
    err = np.linalg.norm(aligned - dst, axis=1)
    ang_err = quat_angle_deg(qe[ie], qg[ig])

    result["ate_rmse"] = float(np.sqrt((err ** 2).mean()))
    result["ate_mean"] = float(err.mean())
    result["ate_median"] = float(np.median(err))
    result["ate_max"] = float(err.max())
    result["rot_rmse_deg"] = float(np.sqrt((ang_err ** 2).mean()))
    return result
