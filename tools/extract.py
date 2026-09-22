#!/usr/bin/env python3
"""GrandTour mission bag -> FAST-LIVO2 offline input.

Usage (container): python3 tools/extract.py <mission_dir> [--out DIR] [--cam front_center|none]
Output <mission_dir>/comfort_offline/
  imu.txt          t_ns wx wy wz ax ay az            (/boxi/stim320/imu)
  lidar/<t_ns>.bin XT32 26B: x y z f32, intensity f32, ring u16, timestamp f64 (absolute seconds)
  cam/<t_ns>.jpg|png  compressed bytes as is              (/boxi/alphasense/<cam>/image_raw/compressed)
  gt.tum           t x y z 0 0 0 1                    (ap20 prism, position only)
  calib.json       T_imu_lidar, T_cam_lidar, T_imu_prism (4x4), cam {frame,model,width,height,K,D}
  manifest.json    message counts and time range
  --livox-only     livox/<t_ns>.bin (same XT32 layout, raw Livox frame, ring=200) + T_lidar_livox (tf_static) added to calib.json.
                   Existing extracts are left untouched. Input for core merging in se3-lio (run_se3lio.sh --lidar multi).
  --cams-only A,B  extract cam_<name>/ per camera in addition and add calib.json cams[name] = {frame,model,width,height,K,D,T_cam_lidar}.
                   front_center keeps the existing cam/ and only fills the calib entry (3-camera livo, run_se3lio.sh --cams).
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
        sys.exit(f'{suffix}.bag missing: {mdir}')
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
    """T_src_dst: dst-frame coordinates -> src-frame coordinates (BFS, both directions)."""
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
    sys.exit(f'no tf path: {src} -> {dst}')


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
        assert t > last, f'lidar stamp goes backwards {last} -> {t}'
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
        assert t > last, f'livox stamp goes backwards {last} -> {t}'
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
        sys.exit(f'{info_topic} missing')
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
    ap.add_argument('--cam', default='front_center', help='alphasense camera name, none to skip')
    ap.add_argument('--livox-only', action='store_true', help='extract only livox/ in addition (existing extract kept)')
    ap.add_argument('--cams-only', metavar='A,B', help='extract cam_<name>/ per camera in addition + calib cams[name] (existing extract kept)')
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
                info, n = dict(calib['cam']), man.get('cam') or len(os.listdir(os.path.join(out, 'cam')))   # old extracts have no cam count in the manifest
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
    if os.path.isdir(os.path.join(out, 'livox')):   # a re-extract must not drop what --livox-only added
        calib['T_lidar_livox'] = chain(edges, LIDAR_FRAME, 'livox_lidar').tolist()
    man = {'mission': os.path.basename(mdir), 'imu': extract_imu(mdir, out)}
    print('imu', man['imu'], flush=True)
    man['lidar'], man['lidar_unsorted_scans'] = extract_lidar(mdir, out)
    print('lidar', man['lidar'], 'unsorted', man['lidar_unsorted_scans'], flush=True)
    man['gt'] = extract_gt(mdir, out) if glob.glob(os.path.join(mdir, '*_ap20_prism_position.bag')) else 0   # Test missions have no GT
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
