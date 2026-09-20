#!/usr/bin/env python3
"""덤프한 스캔(--dump 의 scans/*.pcd)과 LIO 궤적을 OnlineHBA 에 순서대로 넣고 finish 해 HBA 궤적을 낸다.
LIO 없이 HBA 만 다시 돌리는 용도. 덤프에는 IMU 가 없어 정지 노드 중력 factor 는 gravity_file=PATH 로만 넣는다.
사용(컨테이너): python3 tools/hba_offline.py <seq>_imu.tum <scans dir> <out.tum> [K=V,...]   값은 se3lio_run.py --hba 와 같다
"""
import os, sys, time
import numpy as np

from se3_lio.pipeline import OdometryPipeline
from se3_lio.pybind.se3_lio_pybind import _HBAParams, _OnlineHBA


def read_pcd(path):
    with open(path, 'rb') as f:
        n = 0
        while True:
            line = f.readline().decode()
            if line.startswith('POINTS'):
                n = int(line.split()[1])
            if line.startswith('DATA'):
                break
        return np.fromfile(f, np.float32, n * 4).reshape(n, 4)[:, :3]


def main():
    tum, scans, out = sys.argv[1:4]
    hp = _HBAParams()
    for kv in filter(None, (sys.argv[4] if len(sys.argv) > 4 else '').split(',')):
        k, v = kv.split('=')
        cur = getattr(hp, k)
        setattr(hp, k, v if isinstance(cur, str) else [float(x) for x in v.split(':')] if isinstance(cur, list) else type(cur)(float(v)))
    rows = np.loadtxt(tum)
    files = sorted(os.listdir(scans))
    if os.environ.get('HBA_MAX'):
        rows, files = rows[:int(os.environ['HBA_MAX'])], files[:int(os.environ['HBA_MAX'])]
    assert len(rows) == len(files), f'{len(rows)} poses vs {len(files)} scans'
    hba = _OnlineHBA(hp)
    t0 = time.time()
    nan3, zero3 = np.full(3, np.nan), np.zeros(3)
    pace = float(os.environ.get('HBA_PACE', '0'))   # 초/스캔: 0.1 이면 10 Hz 로 넣어 실시간처럼(큐가 안 쌓인다), 0 이면 최대 속도
    for r, f in zip(rows, files):
        t = time.time()
        hba.push(r[4:8], r[1:4], read_pcd(os.path.join(scans, f)), float(r[0]), nan3, zero3, zero3)
        if pace: time.sleep(max(0, pace - (time.time() - t)))
    poses = hba.finish()
    st = np.array(hba.stats()).reshape(-1, 8)   # round, start_scan, kfs, iters, ba_ms, pgo_ms, done_scan, rss_mb
    p = OdometryPipeline.__new__(OdometryPipeline); p.stamps = rows[:, 0]; p.poses = None
    p.save_tum(out, poses)
    hwm = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]
    print(f'rounds {len(st)} (used {(st[:, 6] >= 0).sum()}), ba s last {st[-1, 4] / 1e3 if len(st) else 0:.1f}, rss GB at rounds {st[0, 7] / 1024:.1f}/{st[len(st) // 2, 7] / 1024:.1f}/{st[-1, 7] / 1024:.1f}, '
          f'discarded wait {hba.pending_ms() / 1e3:.1f} s, total {time.time() - t0:.0f} s, peak RSS {int(hwm) / 1e6:.1f} GB -> {out}')


if __name__ == '__main__':
    main()
