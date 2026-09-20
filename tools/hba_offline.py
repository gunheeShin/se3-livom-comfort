#!/usr/bin/env python3
"""덤프한 스캔(--dump 의 scans/*.pcd)과 LIO 궤적을 OnlineHBA 에 순서대로 넣고 finish 해 HBA 궤적을 낸다.
LIO 없이 HBA 만 다시 돌리는 용도이고, hku-mars HBA(run_hba.sh) 출력과 같아야 한다.
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
        setattr(hp, k, type(getattr(hp, k))(float(v)))
    rows = np.loadtxt(tum)
    files = sorted(os.listdir(scans))
    assert len(rows) == len(files), f'{len(rows)} poses vs {len(files)} scans'
    hba = _OnlineHBA(hp)
    t0 = time.time()
    for r, f in zip(rows, files):
        hba.push(r[4:8], r[1:4], read_pcd(os.path.join(scans, f)))
    poses = hba.finish()
    st = np.array(hba.stats())
    p = OdometryPipeline.__new__(OdometryPipeline); p.stamps = rows[:, 0]; p.poses = None
    p.save_tum(out, poses)
    for l in sorted(set(st[:, 0].astype(int))):
        m = st[st[:, 0] == l, 2]
        print(f'layer {l}: {len(m)} windows, ms mean {m.mean():.0f} p95 {np.percentile(m, 95):.0f} max {m.max():.0f}')
    hwm = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]
    print(f'finish {hba.finish_ms() / 1e3:.1f} s, total {time.time() - t0:.0f} s, peak RSS {int(hwm) / 1e6:.1f} GB -> {out}')


if __name__ == '__main__':
    main()
