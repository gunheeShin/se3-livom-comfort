#!/usr/bin/env python3
"""comfort_offline/ → SE(3)-LIO → IMU 프레임 TUM. extrinsic 은 calib.json T_imu_lidar.

입력은 두 가지다. 같은 미션이면 두 방식의 궤적은 비트 단위로 같다(프레임 규칙 comfort_offline.epoch_frames 를 같이 쓴다).
  offline     tools/extract.py 가 풀어 둔 comfort_offline/ 을 읽어 되는 대로 빨리 돈다 — 개발·스윕·제출 궤적.
  online-bag  --online-bag: 미션 폴더의 bag 을 직접 읽어 센서 시각에 맞춰 1배속으로 흘린다(추출 불필요, 프레임을 버리지 않는다).
              <offline_dir> 자리에 미션 폴더, <lidar_dir> 자리에 - 를 준다. calib 은 bag 의 /tf_static·camera_info 에서 만들어
              <out> 옆 calib.json 으로 남기고, 이미지 시각 대비 pose 가 나온 지연을 latency.csv 로 남긴다. 카메라 1대 · Hesai(+Livox).
사용(컨테이너): python3 tools/se3lio_run.py <offline_dir> <lidar_dir> <out_imu.tum> [--config src/se3-lio/config/comfort.yaml]
                                          [--imu-dt S] [--max-frames N] [--set KEY=VAL ...] [--livox-dir DIR]
                                          [--cams A,B,...] [--img-time-offset S] [--dump DIR] [--viz OUT.npz]
--livox-dir: Hesai(lidar_dir) + Livox(livox_dir) 를 따로 읽어 코어가 병합한다(SE3_LIO::mergeLiDARs).
             extrinsic 은 T_imu_lidar(Hesai) 와 T_imu_lidar · dT · T_lidar_livox(tools/common_dT.npy 정련) 두 개.
--cams:      카메라 이름들(기본 front_center, none 이면 끔. front_center 는 cam/, 나머지는 cam_<name>/, extract --cams-only). 첫 카메라의 이미지 스탬프가
             에폭 경계가 되고 pose 스탬프 = 이미지 시각(FAST-LIVO2 방식). config visual.enable 이 켜져 있으면 사진도 읽어
             카메라마다 차례로 photometric 갱신(K·D·T_cam_lidar 는 calib.json cams[name]).
"""
import argparse, json, os
import numpy as np

from se3_lio.config import CameraConfig, load_node_params
from se3_lio.datasets import ComfortOfflineDataset
from se3_lio.pipeline import OdometryPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FrontColor:
    """--viz + --cams: 키프레임 점에 front_center 픽셀 색을 입힌다(화면 밖·카메라 뒤는 회색). 에폭 끝 = 이미지 시각이라 보간이 없다."""

    def __init__(self, cam_dir, c, T_cam_imu, img_time_offset):
        import cv2
        self.cv2, self.dir, self.T, self.off = cv2, cam_dir, T_cam_imu, img_time_offset
        self.files = sorted(os.listdir(cam_dir))
        self.ns = np.array([int(f.split('.')[0]) for f in self.files])
        self.K, self.D = np.array(c['K'], float).reshape(3, 3), np.array(c['D'], float)
        self.project = cv2.fisheye.projectPoints if c['model'] == 'equidistant' else cv2.projectPoints

    def __call__(self, stamp, pts):
        rgb = np.full((len(pts), 3), 200, np.uint8)
        img = self.cv2.imread(os.path.join(self.dir, self.files[np.abs(self.ns - (stamp - self.off) * 1e9).argmin()]))
        pc = pts @ self.T[:3, :3].T + self.T[:3, 3]
        front = np.flatnonzero(pc[:, 2] > 0.1)
        uv = self.project(pc[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)
        u, v = np.round(uv).astype(int).T
        ok = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
        rgb[front[ok]] = img[v[ok], u[ok], ::-1]
        return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('offline_dir')
    ap.add_argument('lidar_dir')
    ap.add_argument('out')
    ap.add_argument('--config', default=os.path.join(ROOT, 'src/se3-lio/config/comfort.yaml'))
    ap.add_argument('--online-bag', action='store_true', help='offline_dir = 미션 폴더(bag), 1배속 재생. lidar_dir 은 안 쓴다(-)')
    ap.add_argument('--imu-dt', type=float, default=0.0)
    ap.add_argument('--max-frames', type=int)
    ap.add_argument('--livox-dir', help='comfort_offline/livox — 주면 코어 내부 병합')
    ap.add_argument('--cams', default='front_center', help='카메라 이름 목록(쉼표) — 이미지 시각이 에폭 경계. none 이면 카메라 없이 LiDAR 스캔 시각')
    ap.add_argument('--img-time-offset', type=float, default=0.0)
    ap.add_argument('--viz', metavar='OUT.npz', help='시각화 덤프 — 궤적·키프레임(0.5 m) 스캔 1/5. 호스트에서 tools/viz_rrd.py 가 rerun .rrd 로 바꾼다')
    ap.add_argument('--dump', metavar='DIR', help='디스큐 스캔(body 프레임, 다운샘플 전)을 DIR/00000.pcd… 로 저장 — 궤적 행과 1:1, HBA 입력')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VAL', help='SE3LIOConfig 필드 덮어쓰기 (voxel_map_plane_thres=1e-3, 리스트는 lidar_range_noises=0.02,0.05)')
    ap.add_argument('--hba', nargs='?', const='', metavar='K=V,...', help='온라인 HBA 백엔드: 키프레임(25스캔) every 개마다 지금까지 전부를 LIO pose 에서 global BA 하고 iSAM2 PGO 에 넣는다. '
                    '끝에는 아무것도 안 돌리고 그 시점의 추정치가 <seq>_hba.tum, 회차 기록 hba_rounds.csv. 값은 _HBAParams 덮어쓰기'
                    '(voxel_size=1.0,downsample_size=0.2,eigen_ratio=0.01,reject_ratio=0.05,max_iter=10,layers=3,threads=8,every=4,hess_const=a:b:c:d:e:f,gravity_sigma_deg=0.02,gravity_file=PATH). 중력 factor 는 정지 토막(2 s 속도 < 2 cm/s, 4 s)의 가속도계 평균을 LIO 의 bias·중력 방향 기준으로 정지 노드에 건다(0 이면 끔, gravity_file 이 있으면 그 파일)')
    a = ap.parse_args()

    params = load_node_params(a.config)
    if a.online_bag:
        from se3_lio.datasets.comfort_bag import ComfortBagDataset, read_calib
        assert ',' not in a.cams, 'online-bag: Hesai(+Livox) + 카메라 1대까지'
        calib = read_calib(a.offline_dir, None if a.cams == 'none' else a.cams)
        json.dump(calib, open(os.path.join(os.path.dirname(a.out), 'calib.json'), 'w'), indent=1)
    else:
        calib = json.load(open(os.path.join(a.offline_dir, 'calib.json')))
    extrinsic = np.array(calib['T_imu_lidar'])
    if a.livox_dir:
        dT = np.load(os.path.join(ROOT, 'tools', 'common_dT.npy'))
        T_imu_livox = extrinsic @ dT @ np.array(calib['T_lidar_livox'])
        params['config'].lidar_extrinsics = [extrinsic.tolist(), T_imu_livox.tolist()]
    for kv in a.set:   # 라이다 목록(multi 의 [Hesai, Livox] 등)이 채워진 뒤에 덮어써야 lidar_*_noises 를 센서별로 줄 수 있다
        k, v = kv.split('=', 1)
        obj = params['config']
        while '.' in k:   # visual.img_point_cov 처럼 중첩 필드
            head, k = k.split('.', 1)
            obj = getattr(obj, head)
        cur = getattr(obj, k)
        setattr(obj, k, [float(x) for x in v.split(',')] if isinstance(cur, list) else type(cur)(float(v)))
    cfg, cam_dirs, cam_calibs = params['config'], None, None
    names = [] if a.cams == 'none' else a.cams.split(',')
    if not names:
        cfg.visual_en = False
    else:
        cam_dirs = [os.path.join(a.offline_dir, 'cam' if n == 'front_center' else f'cam_{n}') for n in names]
        if cfg.visual_en:
            cam_calibs = [calib['cams'][n] for n in names]
            for c in cam_calibs:
                K = np.array(c['K']).reshape(3, 3)
                cfg.cameras.append(CameraConfig(
                    width=c['width'], height=c['height'], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                    dist=[], fisheye=c['model'] == 'equidistant',   # 리더가 undistort 해서 넘긴다
                    T_cam_imu=(np.array(c['T_cam_lidar']) @ np.linalg.inv(extrinsic)).tolist()))
    if a.online_bag:
        assert names, 'online-bag: 이미지 시각이 에폭 경계다(--cams none 은 offline 만)'
        dataset = ComfortBagDataset(a.offline_dir, params['min_range'], a.imu_dt, a.max_frames, params['point_filter_num'],
                                    livox=bool(a.livox_dir), cam=names[0], img_time_offset=a.img_time_offset,
                                    cam_calib=cam_calibs[0] if cam_calibs else None)
    else:
        dataset = ComfortOfflineDataset(a.offline_dir, a.lidar_dir, params['min_range'], a.imu_dt, a.max_frames,
                                        params['point_filter_num'], livox_dir=a.livox_dir, cam_dirs=cam_dirs,
                                        img_time_offset=a.img_time_offset, cam_calibs=cam_calibs)
    hba = None
    if a.hba is not None:
        from se3_lio.pybind.se3_lio_pybind import _HBAParams, _OnlineHBA
        hp = _HBAParams()
        for kv in filter(None, a.hba.split(',')):
            k, v = kv.split('=')
            cur = getattr(hp, k)
            setattr(hp, k, v if isinstance(cur, str) else [float(x) for x in v.split(':')] if isinstance(cur, list) else type(cur)(float(v)))
        if hp.dump_dir:
            os.makedirs(hp.dump_dir, exist_ok=True)
        hba = _OnlineHBA(hp)
    pipeline = OdometryPipeline(dataset, params['config'], extrinsic, hba=hba)
    logger = None
    if a.viz:
        class _VizDump:   # rerun 0.33 은 이미지(py3.8)에 못 올린다 — npz 로 남기고 호스트 tools/viz_rrd.py 가 rrd 로 바꾼다
            def __init__(self):
                self.grav, self.last, self.kf, self.clouds, self.rgbs = None, None, [], [], []
                self.color = None
                if 'front_center' in names:
                    c = calib['cams']['front_center']
                    self.color = _FrontColor(cam_dirs[names.index('front_center')], c,
                                             np.array(c['T_cam_lidar']) @ np.linalg.inv(extrinsic), a.img_time_offset)

            def log_frame(self, stamp, pose, pts, grav=None):   # pts: 코어가 돌려준 디스큐 body 프레임 클라우드
                if self.grav is None:
                    self.grav = np.asarray(grav, float)
                p = pose[:3, 3]
                if self.last is None or np.linalg.norm(p - self.last) >= 0.5:
                    self.last = p.copy()
                    self.kf.append(len(pipeline.poses) - 1)
                    pts = np.asarray(pts, np.float32)[::5, :3]
                    if single:   # 단일 LiDAR 는 원시 lidar 프레임 점이 온다 — body 로
                        pts = pts @ extrinsic[:3, :3].T.astype(np.float32) + extrinsic[:3, 3].astype(np.float32)
                    self.clouds.append(pts)
                    if self.color:
                        self.rgbs.append(self.color(stamp, pts))
        single = not a.livox_dir
        logger = _VizDump()
    pipeline.run(progress=False, dump_dir=a.dump, logger=logger)
    pipeline.save_tum(a.out)
    pipeline.save_timing(os.path.join(os.path.dirname(a.out), 'timing.csv'))
    hwm = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]
    print(f'peak RSS {int(hwm) / 1e6:.2f} GB')   # 이 프로세스(LIO + HBA 워커) 최고 상주 메모리
    if hba is not None:
        assert a.out.endswith('_imu.tum')
        pipeline.save_tum(a.out[:-len('_imu.tum')] + '_hba.tum', pipeline.poses_hba)
        st = np.array(hba.stats()).reshape(-1, 7)   # round, start_scan, kfs, iters, ba_ms, pgo_ms, done_scan(-1: 끝에서 버림)
        np.savetxt(os.path.join(os.path.dirname(a.out), 'hba_rounds.csv'), st, fmt=['%d', '%d', '%d', '%d', '%.1f', '%.1f', '%d'], delimiter=',',
                   header='round,start_scan,kfs,iters,ba_ms,pgo_ms,done_scan', comments='')
        used = st[st[:, 6] >= 0]
        gaps = np.diff(st[:, 1]) if len(st) > 1 else np.zeros(1)
        print(f'hba: rounds {len(st)} (used {len(used)}), kfs last {int(st[-1, 2]) if len(st) else 0}, ba s first/last '
              f'{st[0, 4] / 1e3 if len(st) else 0:.2f}/{st[-1, 4] / 1e3 if len(st) else 0:.2f}, pgo ms max {used[:, 5].max() if len(used) else 0:.0f}, '
              f'round gap scans med/max {np.median(gaps):.0f}/{gaps.max():.0f}, discarded wait {hba.pending_ms() / 1e3:.1f} s')
    if a.online_bag:   # 지연 = pose 가 나온 시각 - 그 이미지가 찍힌 시각(재생 시계). LiDAR 스캔이 끝나길 기다리는 몫(<= 0.1 s) 포함
        lag = np.array([dataset.lag(t, w) for t, w in zip(pipeline.stamps, pipeline.done)]) * 1e3
        np.savetxt(os.path.join(os.path.dirname(a.out), 'latency.csv'), np.c_[pipeline.stamps, lag], fmt=['%.9f', '%.3f'], delimiter=',', header='stamp,lag_ms', comments='')
        print(f'online-bag lag ms: mean {lag.mean():.1f}, p95 {np.percentile(lag, 95):.1f}, max {lag.max():.1f}, last 100 mean {lag[-100:].mean():.1f}')
    if a.viz:
        np.savez(a.viz, stamps=np.array(pipeline.stamps), poses=np.array(pipeline.poses), grav=logger.grav, kf=np.array(logger.kf),
                 kf_len=np.array([len(c) for c in logger.clouds]), pts=np.concatenate(logger.clouds),
                 **({'rgb': np.concatenate(logger.rgbs)} if logger.rgbs else {}))
    print(pipeline.summary())
    print(f'[se3lio] done frames={len(pipeline.poses)} -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
