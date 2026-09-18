#!/usr/bin/env python3
"""IMU 프레임 TUM 궤적 → prism 프레임(제출 형식). T_world_prism = T_world_imu · (T_imu_prism + Δl).
사용: python3 tools/to_prism.py <calib.json> <in_imu.tum> <out_prism.tum> [--grid 0.005]
입력에 중복·역행 타임스탬프가 있으면 마지막 값을 남기고 정렬한다.
--grid S: 레버암 적용 뒤 S초 격자로 보간해 낸다(위치 선형·자세 SLERP, 0.15 s 넘는 공백은 잇지 않음).
"""
import json, sys
import numpy as np


def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, a):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    c = np.clip(np.dot(q0, q1), -1, 1)
    th = np.arccos(c)
    if th < 1e-6:
        q = (1 - a) * q0 + a * q1
    else:
        q = np.sin((1 - a) * th) / np.sin(th) * q0 + np.sin(a * th) / np.sin(th) * q1
    return q / np.linalg.norm(q)


def regrid(t, P, Q, step, max_gap=0.15):
    g = np.arange(t[0], t[-1] + 1e-9, step)
    j = np.clip(np.searchsorted(t, g, side='right') - 1, 0, len(t) - 2)
    a = (g - t[j]) / (t[j + 1] - t[j])
    ok = (t[j + 1] - t[j] <= max_gap) | (a <= 0)
    out = []
    for gi, ji, ai in zip(g[ok], j[ok], a[ok]):
        out.append((gi, (1 - ai) * P[ji] + ai * P[ji + 1], slerp(Q[ji], Q[ji + 1], ai)))
    return out


def main():
    calib, src, dst = sys.argv[1:4]
    grid = float(sys.argv[sys.argv.index('--grid') + 1]) if '--grid' in sys.argv else None
    # Δl: 로컬 Val 6미션 잔차 회귀로 잰 프리즘 lever arm 보정(IMU 프레임). 시각 시프트로 보이던 +11 ms의 정체.
    # 근거: 45 COMFORT/logs/2026-09-10 시간 정합.html
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
    out = regrid(t, P, Q, grid) if grid else zip(t, P, Q)
    n = 0
    with open(dst, 'w') as f:
        for ti, pos, q in out:
            f.write(f'{ti:.6f} ' + ' '.join(f'{x:.6f}' for x in pos) + ' ' + ' '.join(f'{x:.8f}' for x in q) + '\n')
            n += 1
    print(f'{len(rows)} poses -> {n} -> {dst}')


if __name__ == '__main__':
    main()
