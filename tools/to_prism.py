#!/usr/bin/env python3
"""IMU-frame TUM trajectory -> prism frame (submission format). T_world_prism = T_world_imu * (T_imu_prism + dl).
Usage: python3 tools/to_prism.py <calib.json> <in_imu.tum> <out_prism.tum>
Duplicate or backwards timestamps in the input keep the last value and are sorted.
"""
import json, sys
import numpy as np


def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    calib, src, dst = sys.argv[1:4]
    # dl: prism lever-arm correction (IMU frame) fitted by residual regression on the 6 local Val missions. It is what looked like a +11 ms time shift.
    # Basis: 45 COMFORT/logs/2026-09-10 time alignment note
    t_ip = np.array(json.load(open(calib))['T_imu_prism'])[:3, 3] + np.array([-0.0057, 0.0018, -0.0114])
    rows = {}
    for line in open(src):
        p = line.split()
        if len(p) == 8:
            rows[float(p[0])] = np.array(p[1:], dtype=float)
    t = np.array(sorted(rows))
    V = np.array([rows[k] for k in t])
    P = V[:, :3] + np.einsum('nij,j->ni', np.array([quat_to_R(q) for q in V[:, 3:]]), t_ip)
    Q = V[:, 3:] / np.linalg.norm(V[:, 3:], axis=1, keepdims=True)
    n = 0
    with open(dst, 'w') as f:
        for ti, pos, q in zip(t, P, Q):
            f.write(f'{ti:.6f} ' + ' '.join(f'{x:.6f}' for x in pos) + ' ' + ' '.join(f'{x:.8f}' for x in q) + '\n')
            n += 1
    print(f'{len(rows)} poses -> {n} -> {dst}')


if __name__ == '__main__':
    main()
