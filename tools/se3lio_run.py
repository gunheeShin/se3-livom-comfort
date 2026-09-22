#!/usr/bin/env python3
"""comfort_offline/ -> SE(3)-LIO -> TUM in the IMU frame. extrinsic is calib.json T_imu_lidar.

Two inputs. For the same mission both give bit-identical trajectories (they share the frame rule comfort_offline.epoch_frames).
  offline     reads the comfort_offline/ that tools/extract.py wrote and runs as fast as it can: development, sweeps, submission trajectories.
  online-bag  --online-bag: reads the mission folder's bags directly and streams them at 1x by sensor time (no extraction, no frame dropped).
              Give the mission folder as <offline_dir> and - as <lidar_dir>. calib is built from the bag's /tf_static and camera_info and
              written as calib.json next to <out>; the pose delay vs image time goes to latency.csv. One camera, Hesai (+Livox).
Usage (container): python3 tools/se3lio_run.py <offline_dir> <lidar_dir> <out_imu.tum> [--config src/se3-lio/config/comfort.yaml]
                                          [--imu-dt S] [--max-frames N] [--set KEY=VAL ...] [--livox-dir DIR]
                                          [--cams A,B,...] [--img-time-offset S] [--backend K=V,...]
--livox-dir: read Hesai (lidar_dir) and Livox (livox_dir) separately and let the core merge them (SE3_LIO::mergeLiDARs).
             Two extrinsics: T_imu_lidar (Hesai) and T_imu_lidar * dT * T_lidar_livox (dT refined in tools/common_dT.npy).
--cams:      camera names (default front_center, none = off. front_center is cam/, others cam_<name>/, extract --cams-only). The first camera's image stamps
             become the epoch boundaries and pose stamp = image time (FAST-LIVO2 style). If config visual.enable is on the images are read too and
             each camera runs a photometric update in turn (K, D, T_cam_lidar from calib.json cams[name]).
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
    ap.add_argument('--online-bag', action='store_true', help='offline_dir = mission folder (bag), 1x playback. lidar_dir is unused (-)')
    ap.add_argument('--imu-dt', type=float, default=0.0)
    ap.add_argument('--max-frames', type=int)
    ap.add_argument('--livox-dir', help='comfort_offline/livox: if given, the core merges the two LiDARs')
    ap.add_argument('--cams', default='front_center', help='camera names (comma separated); image times are the epoch boundaries. none = no camera, LiDAR scan times instead')
    ap.add_argument('--img-time-offset', type=float, default=0.0)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VAL', help='override an SE3LIOConfig field (voxel_map_plane_thres=1e-3; lists as lidar_range_noises=0.02,0.05)')
    ap.add_argument('--backend', nargs='?', const='', metavar='K=V,...', help='online backend: every `every` keyframes (25 scans each) a global plane BA over everything so far from the LIO poses, fed into an iSAM2 pose graph. '
                    'Nothing runs at the end; the estimate at that moment goes to <seq>_backend.tum, rounds to backend_rounds.csv. Values override _BackendParams'
                    '(voxel_size=1.0,downsample_size=0.2,eigen_ratio=0.01,reject_ratio=0.05,max_iter=10,layers=3,threads=8,every=4,hess_const=a:b:c:d:e:f,gravity_sigma_deg=0.02). The gravity factor puts the accelerometer mean of a stationary chunk (speed < 2 cm/s over 2 s, 4 s long) on the stationary node, relative to the LIO bias and gravity direction (0 = off)')
    a = ap.parse_args()

    params = load_node_params(a.config)
    if a.online_bag:
        from se3_lio.datasets.comfort_bag import ComfortBagDataset, read_calib
        assert ',' not in a.cams, 'online-bag: Hesai (+Livox) + at most one camera'
        calib = read_calib(a.offline_dir, None if a.cams == 'none' else a.cams)
        json.dump(calib, open(os.path.join(os.path.dirname(a.out), 'calib.json'), 'w'), indent=1)
    else:
        calib = json.load(open(os.path.join(a.offline_dir, 'calib.json')))
    extrinsic = np.array(calib['T_imu_lidar'])
    if a.livox_dir:
        dT = np.load(os.path.join(ROOT, 'tools', 'common_dT.npy'))
        T_imu_livox = extrinsic @ dT @ np.array(calib['T_lidar_livox'])
        params['config'].lidar_extrinsics = [extrinsic.tolist(), T_imu_livox.tolist()]
    for kv in a.set:   # override after the LiDAR list (multi: [Hesai, Livox]) is filled so lidar_*_noises can be given per sensor
        k, v = kv.split('=', 1)
        obj = params['config']
        while '.' in k:   # nested field such as visual.img_point_cov
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
                    dist=[], fisheye=c['model'] == 'equidistant',   # the reader undistorts before handing over
                    T_cam_imu=(np.array(c['T_cam_lidar']) @ np.linalg.inv(extrinsic)).tolist()))
    if a.online_bag:
        assert names, 'online-bag: image times are the epoch boundaries (--cams none is offline only)'
        dataset = ComfortBagDataset(a.offline_dir, params['min_range'], a.imu_dt, a.max_frames, params['point_filter_num'],
                                    livox=bool(a.livox_dir), cam=names[0], img_time_offset=a.img_time_offset,
                                    cam_calib=cam_calibs[0] if cam_calibs else None)
    else:
        dataset = ComfortOfflineDataset(a.offline_dir, a.lidar_dir, params['min_range'], a.imu_dt, a.max_frames,
                                        params['point_filter_num'], livox_dir=a.livox_dir, cam_dirs=cam_dirs,
                                        img_time_offset=a.img_time_offset, cam_calibs=cam_calibs)
    backend = None
    if a.backend is not None:
        from se3_lio.pybind.se3_lio_pybind import _BackendParams, _OnlineBackend
        hp = _BackendParams()
        for kv in filter(None, a.backend.split(',')):
            k, v = kv.split('=')
            cur = getattr(hp, k)
            setattr(hp, k, v if isinstance(cur, str) else [float(x) for x in v.split(':')] if isinstance(cur, list) else type(cur)(float(v)))
        if hp.dump_dir:
            os.makedirs(hp.dump_dir, exist_ok=True)
        backend = _OnlineBackend(hp)
    pipeline = OdometryPipeline(dataset, params['config'], extrinsic, backend=backend)
    pipeline.run(progress=False)
    pipeline.save_tum(a.out)
    pipeline.save_timing(os.path.join(os.path.dirname(a.out), 'timing.csv'))
    hwm = [l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]
    print(f'peak RSS {int(hwm) / 1e6:.2f} GB')   # peak resident memory of this process (LIO + backend worker)
    if backend is not None:
        assert a.out.endswith('_imu.tum')
        pipeline.save_tum(a.out[:-len('_imu.tum')] + '_backend.tum', pipeline.poses_backend)
        st = np.array(backend.stats()).reshape(-1, 8)   # round, start_scan, kfs, iters, ba_ms, pgo_ms, done_scan (-1: discarded at the end), rss_mb
        np.savetxt(os.path.join(os.path.dirname(a.out), 'backend_rounds.csv'), st, fmt=['%d', '%d', '%d', '%d', '%.1f', '%.1f', '%d', '%.0f'], delimiter=',',
                   header='round,start_scan,kfs,iters,ba_ms,pgo_ms,done_scan,rss_mb', comments='')
        used = st[st[:, 6] >= 0]
        gaps = np.diff(st[:, 1]) if len(st) > 1 else np.zeros(1)
        print(f'backend: rounds {len(st)} (used {len(used)}), kfs last {int(st[-1, 2]) if len(st) else 0}, ba s first/last '
              f'{st[0, 4] / 1e3 if len(st) else 0:.2f}/{st[-1, 4] / 1e3 if len(st) else 0:.2f}, pgo ms max {used[:, 5].max() if len(used) else 0:.0f}, '
              f'round gap scans med/max {np.median(gaps):.0f}/{gaps.max():.0f}, discarded wait {backend.pending_ms() / 1e3:.1f} s, '
              f'rss GB at rounds first/last {st[0, 7] / 1024 if len(st) else 0:.1f}/{st[-1, 7] / 1024 if len(st) else 0:.1f}')
    if a.online_bag:   # delay = time the pose came out - time that image was taken (playback clock). Includes waiting for the LiDAR scan to end (<= 0.1 s)
        lag = np.array([dataset.lag(t, w) for t, w in zip(pipeline.stamps, pipeline.done)]) * 1e3
        np.savetxt(os.path.join(os.path.dirname(a.out), 'latency.csv'), np.c_[pipeline.stamps, lag], fmt=['%.9f', '%.3f'], delimiter=',', header='stamp,lag_ms', comments='')
        print(f'online-bag lag ms: mean {lag.mean():.1f}, p95 {np.percentile(lag, 95):.1f}, max {lag.max():.1f}, last 100 mean {lag[-100:].mean():.1f}')
    print(pipeline.summary())
    print(f'[se3lio] done frames={len(pipeline.poses)} -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
