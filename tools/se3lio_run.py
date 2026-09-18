#!/usr/bin/env python3
"""comfort_offline/ → SE(3)-LIO → IMU 프레임 TUM. extrinsic 은 calib.json T_imu_lidar.

사용(컨테이너): python3 tools/se3lio_run.py <offline_dir> <lidar_dir> <out_imu.tum> [--config src/se3-lio/config/comfort.yaml]
                                          [--imu-dt S] [--max-frames N] [--set KEY=VAL ...] [--livox-dir DIR]
                                          [--cams A,B,...] [--img-time-offset S] [--dump DIR] [--viz OUT.npz]
--livox-dir: Hesai(lidar_dir) + Livox(livox_dir) 를 따로 읽어 코어가 병합한다(SE3_LIO::mergeLiDARs).
             extrinsic 은 T_imu_lidar(Hesai) 와 T_imu_lidar · dT · T_lidar_livox(tools/common_dT.npy 정련) 두 개.
--cams:      카메라 이름들(front_center 는 cam/, 나머지는 cam_<name>/, extract --cams-only). 첫 카메라의 이미지 스탬프가
             에폭 경계가 되고 pose 스탬프 = 이미지 시각(FAST-LIVO2 방식). config visual.enable 이 켜져 있으면 사진도 읽어
             카메라마다 차례로 photometric 갱신(K·D·T_cam_lidar 는 calib.json cams[name]).
"""
import argparse, json, os
import numpy as np

from se3_lio.config import CameraConfig, load_node_params
from se3_lio.datasets import ComfortOfflineDataset
from se3_lio.pipeline import OdometryPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('offline_dir')
    ap.add_argument('lidar_dir')
    ap.add_argument('out')
    ap.add_argument('--config', default=os.path.join(ROOT, 'src/se3-lio/config/comfort.yaml'))
    ap.add_argument('--imu-dt', type=float, default=0.0)
    ap.add_argument('--max-frames', type=int)
    ap.add_argument('--livox-dir', help='comfort_offline/livox — 주면 코어 내부 병합')
    ap.add_argument('--cams', help='카메라 이름 목록(쉼표) — 주면 이미지 시각이 에폭 경계')
    ap.add_argument('--img-time-offset', type=float, default=0.0)
    ap.add_argument('--dump', metavar='DIR', help='디스큐 스캔(body 프레임, 다운샘플 전)을 DIR/00000.pcd… 로 저장 — 궤적 행과 1:1, HBA 입력')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VAL', help='SE3LIOConfig 필드 덮어쓰기 (voxel_map_plane_thres=1e-3, 리스트는 lidar_range_noises=0.02,0.05)')
    a = ap.parse_args()

    ap.add_argument('--viz', metavar='OUT.npz', help='시각화 덤프 — 궤적·키프레임(0.5 m) 스캔 1/5. 호스트에서 tools/viz_rrd.py 가 rerun .rrd 로 바꾼다')
    params = load_node_params(a.config)
    for kv in a.set:
        k, v = kv.split('=', 1)
        obj = params['config']
        while '.' in k:   # visual.img_point_cov 처럼 중첩 필드
            head, k = k.split('.', 1)
            obj = getattr(obj, head)
        cur = getattr(obj, k)
        setattr(obj, k, [float(x) for x in v.split(',')] if isinstance(cur, list) else type(cur)(float(v)))
    calib = json.load(open(os.path.join(a.offline_dir, 'calib.json')))
    extrinsic = np.array(calib['T_imu_lidar'])
    if a.livox_dir:
        dT = np.load(os.path.join(ROOT, 'tools', 'common_dT.npy'))
        T_imu_livox = extrinsic @ dT @ np.array(calib['T_lidar_livox'])
        params['config'].lidar_extrinsics = [extrinsic.tolist(), T_imu_livox.tolist()]
    cfg, cam_dirs, cam_calibs = params['config'], None, None
    if a.cams:
        names = a.cams.split(',')
        cam_dirs = [os.path.join(a.offline_dir, 'cam' if n == 'front_center' else f'cam_{n}') for n in names]
        if cfg.visual_en:
            cam_calibs = [calib['cams'][n] for n in names]
            for c in cam_calibs:
                K = np.array(c['K']).reshape(3, 3)
                cfg.cameras.append(CameraConfig(
                    width=c['width'], height=c['height'], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                    dist=[], fisheye=c['model'] == 'equidistant',   # 리더가 undistort 해서 넘긴다
                    T_cam_imu=(np.array(c['T_cam_lidar']) @ np.linalg.inv(extrinsic)).tolist()))
    dataset = ComfortOfflineDataset(a.offline_dir, a.lidar_dir, params['min_range'], a.imu_dt, a.max_frames,
                                    params['point_filter_num'], livox_dir=a.livox_dir, cam_dirs=cam_dirs,
                                    img_time_offset=a.img_time_offset, cam_calibs=cam_calibs)
    pipeline = OdometryPipeline(dataset, params['config'], extrinsic)
    logger = None
    if a.viz:
        class _VizDump:   # rerun 0.33 은 이미지(py3.8)에 못 올린다 — npz 로 남기고 호스트 tools/viz_rrd.py 가 rrd 로 바꾼다
            def __init__(self):
                self.grav, self.last, self.kf, self.clouds = None, None, [], []

            def log_frame(self, stamp, pose, pts, grav=None):   # pts: 코어가 돌려준 디스큐 body 프레임 클라우드
                if self.grav is None:
                    self.grav = np.asarray(grav, float)
                p = pose[:3, 3]
                if self.last is None or np.linalg.norm(p - self.last) >= 0.5:
                    self.last = p.copy()
                    self.kf.append(len(pipeline.poses) - 1)
                    self.clouds.append(np.asarray(pts, np.float32)[::5, :3])
        logger = _VizDump()
    pipeline.run(progress=False, dump_dir=a.dump, logger=logger)
    pipeline.save_tum(a.out)
    pipeline.save_timing(os.path.join(os.path.dirname(a.out), 'timing.csv'))
    print(pipeline.summary())
    print(f'[se3lio] done frames={len(pipeline.poses)} -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
    if a.viz:
        np.savez(a.viz, stamps=np.array(pipeline.stamps), poses=np.array(pipeline.poses), grav=logger.grav, kf=np.array(logger.kf),
                 kf_len=np.array([len(c) for c in logger.clouds]), pts=np.concatenate(logger.clouds))
