#!/usr/bin/env python3
"""GrandTour 미션 bag → FAST-LIVO2 오프라인 입력.

사용(컨테이너): python3 tools/extract.py <mission_dir> [--out DIR] [--cam front_center|none]
출력 <mission_dir>/comfort_offline/
  imu.txt          t_ns wx wy wz ax ay az            (/boxi/stim320/imu)
  lidar/<t_ns>.bin XT32 26B: x y z f32, intensity f32, ring u16, timestamp f64(절대초)
  cam/<t_ns>.jpg|png  압축 바이트 원본                (/boxi/alphasense/<cam>/image_raw/compressed)
  gt.tum           t x y z 0 0 0 1                    (ap20 prism, 위치만)
  calib.json       T_imu_lidar, T_cam_lidar, T_imu_prism (4x4), cam {frame,model,width,height,K,D}
  manifest.json    메시지 수·시간 범위
  --livox-only     livox/<t_ns>.bin (같은 XT32 레이아웃, Livox 프레임 원본, ring=200) + calib.json 에 T_lidar_livox(tf_static) 추가.
                   기존 추출물은 건드리지 않는다. se3-lio 코어 내부 병합(run_se3lio.sh --lidar multi) 입력.
  --cams-only A,B  cam_<name>/ 를 카메라별로 추가 추출하고 calib.json 에 cams[name] = {frame,model,width,height,K,D,T_cam_lidar}.
                   front_center 는 기존 cam/ 을 그대로 쓰고 calib 항목만 채운다(3카메라 livo, run_se3lio.sh --cams).
"""
import argparse, glob, json, os, sys
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

XT32 = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring', 'timestamp'],
                 'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f8'],
                 'offsets': [0, 4, 8, 12, 16, 18], 'itemsize': 26})
PF_DTYPE = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}
IMU_FRAME, LIDAR_FRAME, PRISM_FRAME = 'stim320_imu', 'hesai_lidar', 'prism'


def messages(path, topics):
    with Reader(path) as r:
        ts = get_typestore(Stores.ROS1_NOETIC)
        add = {}
        for c in r.connections:
            d = getattr(c.msgdef, 'data', c.msgdef)
            if d:
                add.update(get_types_from_msg(d, c.msgtype))
        ts.register(add)
        cs = [c for c in r.connections if c.topic in topics]
        for c, t, raw in r.messages(connections=cs):
            yield c.topic, ts.deserialize_ros1(raw, c.msgtype)


def stamp_ns(h):
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def bag(mdir, suffix):
    m = glob.glob(os.path.join(mdir, f'*_{suffix}.bag'))
    if not m:
        sys.exit(f'{suffix}.bag 없음: {mdir}')
    return m[0]


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf_tree(mdir):
    edges = {}
    for _, m in messages(bag(mdir, 'tf_minimal'), ['/tf_static']):
        for tr in m.transforms:
            p, q = tr.transform.translation, tr.transform.rotation
            T = np.eye(4)
            T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
            T[:3, 3] = [p.x, p.y, p.z]
            edges.setdefault((tr.header.frame_id, tr.child_frame_id), T)
    return edges


def chain(edges, src, dst):
    """T_src_dst: dst 프레임 좌표 → src 프레임 좌표 (BFS, 양방향)."""
    adj = {}
    for (a, b), T in edges.items():
        adj.setdefault(a, []).append((b, T))
        adj.setdefault(b, []).append((a, np.linalg.inv(T)))
    seen, q = {src: np.eye(4)}, [src]
    while q:
        a = q.pop(0)
        if a == dst:
            return seen[a]
        for b, T in adj.get(a, []):
            if b not in seen:
                seen[b] = seen[a] @ T
                q.append(b)
    sys.exit(f'tf 경로 없음: {src} -> {dst}')


def extract_imu(mdir, out):
    n = 0
    with open(os.path.join(out, 'imu.txt'), 'w') as f:
        for _, m in messages(bag(mdir, 'stim320_imu'), ['/boxi/stim320/imu']):
            w, a = m.angular_velocity, m.linear_acceleration
            f.write(f'{stamp_ns(m.header)} {w.x!r} {w.y!r} {w.z!r} {a.x!r} {a.y!r} {a.z!r}\n')
            n += 1
    return n


def extract_lidar(mdir, out):
    d = os.path.join(out, 'lidar')
    os.makedirs(d, exist_ok=True)
    n, unsorted, last = 0, 0, -1
    for _, m in messages(bag(mdir, 'hesai'), ['/boxi/hesai/points']):
        src = np.dtype({'names': [f.name for f in m.fields], 'formats': [PF_DTYPE[f.datatype] for f in m.fields],
                        'offsets': [f.offset for f in m.fields], 'itemsize': m.point_step})
        pts = np.frombuffer(m.data, dtype=src, count=m.width * m.height)
        o = np.empty(len(pts), dtype=XT32)
        for k in XT32.names:
            o[k] = pts[k]
        t = stamp_ns(m.header)
        assert t > last, f'lidar stamp 역행 {last} -> {t}'
        last = t
        unsorted += int(np.any(np.diff(o['timestamp']) < 0))
        o.tofile(os.path.join(d, f'{t}.bin'))
        n += 1
    return n, unsorted


def extract_livox(mdir, out):
    d = os.path.join(out, 'livox')
    os.makedirs(d, exist_ok=True)
    n, last = 0, -1
    for _, m in messages(bag(mdir, 'livox'), ['/boxi/livox/points']):
        src = np.dtype({'names': [f.name for f in m.fields], 'formats': [PF_DTYPE[f.datatype] for f in m.fields],
                        'offsets': [f.offset for f in m.fields], 'itemsize': m.point_step})
        pts = np.frombuffer(m.data, dtype=src, count=m.width * m.height)
        ok = np.isfinite(pts['x']) & np.isfinite(pts['y']) & np.isfinite(pts['z'])
        pts = pts[ok]
        o = np.empty(len(pts), dtype=XT32)
        o['x'], o['y'], o['z'], o['intensity'] = pts['x'], pts['y'], pts['z'], pts['intensity']
        o['ring'] = 200
        o['timestamp'] = pts['timestamp'].astype(np.float64) * 1e-9
        o = o[np.argsort(o['timestamp'], kind='stable')]
        t = stamp_ns(m.header)
        assert t > last, f'livox stamp 역행 {last} -> {t}'
        last = t
        o.tofile(os.path.join(d, f'{t}.bin'))
        n += 1
    return n


def extract_cam(mdir, out, cam, subdir='cam'):
    d = os.path.join(out, subdir)
    os.makedirs(d, exist_ok=True)
    img_topic = f'/boxi/alphasense/{cam}/image_raw/compressed'
    info_topic = f'/boxi/alphasense/{cam}/camera_info'
    n, info = 0, None
    for topic, m in messages(bag(mdir, 'alphasense'), [img_topic, info_topic]):
        if topic == info_topic:
            if info is None:
                info = {'frame': m.header.frame_id, 'model': m.distortion_model, 'width': int(m.width),
                        'height': int(m.height), 'K': [float(v) for v in m.K], 'D': [float(v) for v in m.D]}
            continue
        ext = 'png' if 'png' in m.format.lower() else 'jpg'
        with open(os.path.join(d, f'{stamp_ns(m.header)}.{ext}'), 'wb') as f:
            f.write(m.data.tobytes())
        n += 1
    if n and info is None:
        sys.exit(f'{info_topic} 없음')
    return n, info


def extract_gt(mdir, out):
    n = 0
    with Reader(bag(mdir, 'ap20_prism_position')) as r:
        topics = [c.topic for c in r.connections]
    with open(os.path.join(out, 'gt.tum'), 'w') as f:
        for _, m in messages(bag(mdir, 'ap20_prism_position'), topics):
            p = getattr(m, 'point', None) or getattr(m, 'position', None)
            f.write(f'{stamp_ns(m.header) * 1e-9:.6f} {p.x!r} {p.y!r} {p.z!r} 0 0 0 1\n')
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mission_dir')
    ap.add_argument('--out')
    ap.add_argument('--cam', default='front_center', help='alphasense 카메라 이름, none이면 생략')
    ap.add_argument('--livox-only', action='store_true', help='livox/ 만 추가 추출 (기존 추출물 유지)')
    ap.add_argument('--cams-only', metavar='A,B', help='카메라별 cam_<name>/ 추가 추출 + calib cams[name] (기존 추출물 유지)')
    a = ap.parse_args()
    mdir = a.mission_dir.rstrip('/')
    out = a.out or os.path.join(mdir, 'comfort_offline')
    os.makedirs(out, exist_ok=True)

    edges = tf_tree(mdir)
    if a.livox_only:
        calib = json.load(open(os.path.join(out, 'calib.json')))
        calib['T_lidar_livox'] = chain(edges, LIDAR_FRAME, 'livox_lidar').tolist()
        man = json.load(open(os.path.join(out, 'manifest.json')))
        man['livox'] = extract_livox(mdir, out)
        json.dump(calib, open(os.path.join(out, 'calib.json'), 'w'), indent=1)
        json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
        print('livox', man['livox'], 'done', out)
        return
    if a.cams_only:
        calib = json.load(open(os.path.join(out, 'calib.json')))
        man = json.load(open(os.path.join(out, 'manifest.json')))
        calib.setdefault('cams', {})
        for cam in a.cams_only.split(','):
            sub = 'cam' if cam == 'front_center' else f'cam_{cam}'
            if cam == 'front_center' and calib.get('cam'):
                info, n = dict(calib['cam']), man.get('cam') or len(os.listdir(os.path.join(out, 'cam')))   # 옛 추출본은 manifest 에 cam 수가 없다
            else:
                n, info = extract_cam(mdir, out, cam, sub)
                man[sub] = n
            info['T_cam_lidar'] = chain(edges, info['frame'], LIDAR_FRAME).tolist()
            calib['cams'][cam] = info
            print(cam, n, info['frame'], flush=True)
        json.dump(calib, open(os.path.join(out, 'calib.json'), 'w'), indent=1)
        json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
        print('cams done', out)
        return
    calib = {'T_imu_lidar': chain(edges, IMU_FRAME, LIDAR_FRAME).tolist(),
             'T_imu_prism': chain(edges, IMU_FRAME, PRISM_FRAME).tolist(), 'cam': None}
    if os.path.isdir(os.path.join(out, 'livox')):   # 재추출이 --livox-only 가 넣은 항목을 지우지 않게
        calib['T_lidar_livox'] = chain(edges, LIDAR_FRAME, 'livox_lidar').tolist()
    man = {'mission': os.path.basename(mdir), 'imu': extract_imu(mdir, out)}
    print('imu', man['imu'], flush=True)
    man['lidar'], man['lidar_unsorted_scans'] = extract_lidar(mdir, out)
    print('lidar', man['lidar'], 'unsorted', man['lidar_unsorted_scans'], flush=True)
    man['gt'] = extract_gt(mdir, out) if glob.glob(os.path.join(mdir, '*_ap20_prism_position.bag')) else 0   # Test 미션은 GT 없음
    print('gt', man['gt'], flush=True)
    if a.cam != 'none':
        man['cam'], info = extract_cam(mdir, out, a.cam)
        calib['cam'] = info
        calib['T_cam_lidar'] = chain(edges, info['frame'], LIDAR_FRAME).tolist()
        print('cam', man['cam'], info['frame'], info['model'], info['width'], 'x', info['height'], flush=True)
    json.dump(calib, open(os.path.join(out, 'calib.json'), 'w'), indent=1)
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print('done', out)


if __name__ == '__main__':
    main()
