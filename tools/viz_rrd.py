#!/usr/bin/env python3
"""viz.npz(se3lio_run.py --viz) → rerun .rrd. 호스트에서 돈다(rerun-sdk 0.33.1, 뷰어도 같은 버전: rerun viz.rrd).

사용: python3 tools/viz_rrd.py results/<name>/viz.npz [out.rrd]
"""
import os, sys, types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType('se3_lio')   # se3_lio/__init__ 의 pybind import 를 건너뛴다(호스트엔 바인딩이 없다)
pkg.__path__ = [os.path.join(ROOT, 'src/se3-lio/python/se3_lio')]
sys.modules['se3_lio'] = pkg
from se3_lio.viz.rerun_logger import RerunLogger

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + '.rrd'
d = np.load(src)
ends = np.cumsum(d['kf_len'])
clouds = dict(zip(d['kf'].tolist(), np.split(d['pts'], ends[:-1])))
rgbs = dict(zip(d['kf'].tolist(), np.split(d['rgb'], ends[:-1]))) if 'rgb' in d else {}
logger = RerunLogger(np.eye(4), keyframe_dist=0.0, save_path=out)   # 키프레임은 이미 골라 왔다 — 빈 스캔은 로거가 건너뛴다
VOX, has_color, occupied = 0.05, set(), set()


def fresh(pts, T, hit):
    """world 복셀마다 색 우선: 색 점은 색이 없던 복셀에만, 회색 점은 빈 복셀에만 내보낸다(이미 나간 회색은 못 지운다)."""
    ijk = np.floor((pts @ T[:3, :3].T + T[:3, 3]) / VOX).astype(np.int64)
    keys = ((ijk[:, 0] << 42) + (ijk[:, 1] << 21) + ijk[:, 2]).tolist()
    keep = np.zeros(len(pts), bool)
    for j in np.argsort(~hit, kind='stable').tolist():   # 같은 키프레임 안에서도 색 점 먼저
        k = keys[j]
        if k in (has_color if hit[j] else occupied):
            continue
        occupied.add(k)
        if hit[j]:
            has_color.add(k)
        keep[j] = True
    return keep


n_out = 0
for i, (t, T) in enumerate(zip(d['stamps'], d['poses'])):
    pts, rgb, hit = clouds.get(i, np.empty((0, 3))), rgbs.get(i), None
    if rgb is not None:
        hit = (rgb != 200).any(1)   # 200 = 러너가 채운 '색 없음'
        keep = fresh(pts, T, hit)
        pts, rgb, hit = pts[keep], rgb[keep], hit[keep]
        n_out += len(pts)
    logger.log_frame(t, T, pts, d['grav'], rgb, hit)
print(f'{out}: {len(d["stamps"])} poses, {len(clouds)} keyframes, {len(d["pts"])} points' + (f' -> {n_out} after voxel filter' if rgbs else ''))
