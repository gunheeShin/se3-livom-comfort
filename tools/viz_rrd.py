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
logger = RerunLogger(np.eye(4), keyframe_dist=0.0, save_path=out)   # 키프레임은 이미 골라 왔다 — 빈 스캔은 로거가 건너뛴다
for i, (t, T) in enumerate(zip(d['stamps'], d['poses'])):
    logger.log_frame(t, T, clouds.get(i, np.empty((0, 3))), d['grav'])
print(f'{out}: {len(d["stamps"])} poses, {len(clouds)} keyframes, {len(d["pts"])} points')
