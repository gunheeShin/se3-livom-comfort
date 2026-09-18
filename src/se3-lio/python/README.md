# SE(3)-LIO Python bindings

Python bindings for the ROS-agnostic SE(3) LiDAR-Inertial Odometry C++ core.

## Install

Linux x86_64 wheels are on PyPI (the C++ core is bundled in):

```bash
pip install se3-lio          # add [viz] for the live Polyscope viewer
```

## Build from source

The bindings link the C++ core, which needs OpenMP (libgomp); Eigen and Sophus
are fetched and compiled in. Build inside the project's ROS2 Docker image
(`docker/ros2/`), where OpenMP is available.

```bash
# inside the container, from the repo root
pip install --no-build-isolation -ve ./python/   # requires: pip install pybind11 scikit-build-core
# or, with build isolation (pulls build deps automatically):
pip install -ve ./python/
```

## Usage (high-level API)

```python
import numpy as np
from se3_lio import SE3LIO, SE3LIOConfig, load_node_params

config = SE3LIOConfig(downsample_resolution=0.5, max_iter=4, verbose=False)
odom = SE3LIO(config, lidar_extrinsic=np.eye(4))

# or load the ROS2 node's params.yaml (same mapping as lio_node.cpp):
params = load_node_params("config/ntu.yaml")
odom = SE3LIO(params["config"], params["extrinsic"])

# points:      (N, 3) float xyz in the LiDAR frame
# point_times: (N,)   float per-point time offset from frame start [s]
# imu:         (M, 7) float rows of [t, ax, ay, az, gx, gy, gz]
state, cloud = odom.register_frame(points, point_times, imu, frame_stamp)
print(state.pose, state.vel, state.covariance)   # cloud: (N,3) deskewed, body frame
```

`register_frame` mirrors the per-frame data path of the ROS2 node: it builds a
synced measurement, applies the LiDAR extrinsic, sorts points by relative
timestamp, and runs one `estimatePose` step. Set `verbose=True` to enable the
core's per-frame stdout logging (off by default).

## CLI pipeline

`se3_lio_pipeline` runs odometry over a rosbag and writes a TUM trajectory — the
Python equivalent of the node, offline. The input format is auto-detected:

| input | format | LiDAR | reader |
|-------|--------|-------|--------|
| rosbag2 directory | ROS2 | Livox `CustomMsg` | `rosbag2_py` |
| `*.bag` file | ROS1 | Ouster `PointCloud2` | `rosbags` |

```bash
# ROS2 / Livox (rosbag2 directory)
se3_lio_pipeline <rosbag_dir> --config my_rig.yaml --imu-topic /livox/imu --lidar-topic /livox/lidar \
    --max-frames 300 --output results/

# ROS1 / Ouster (e.g. NTU VIRAL .bag)
se3_lio_pipeline eee_01.bag --config config/ntu.yaml \
    --max-frames 1500 --output results/
# -> results/<bag>_se3lio.tum  + a summary (frames, path length, final pose)
```

`--config` is a small YAML giving the LiDAR→IMU **extrinsic** (and, optionally,
topics + algorithm overrides); keys you omit fall back to the
[`SE3LIOConfig`](se3_lio/config.py) code defaults. The repo's `config/*.yaml`
cover our datasets — for your own rig, write just the extrinsic and pass the
topics on the CLI:

```bash
# my_rig.yaml -> sensors: { t_exts: [x,y,z], q_exts: [w,x,y,z] }
se3_lio_pipeline my.bag --config my_rig.yaml \
    --imu-topic /my/imu --lidar-topic /my/points
```

Use `--input-type ros2-livox|ros1-ouster` to override auto-detection. It reads
the bag, reproduces the node's conversion + synchronization
([rosbag.py](se3_lio/datasets/rosbag.py) / [ros1bag.py](se3_lio/datasets/ros1bag.py)),
and runs [OdometryPipeline](se3_lio/pipeline.py) over the frames. Use the same
pieces programmatically:

```python
from se3_lio import load_node_params
from se3_lio.datasets import RosbagDataset
from se3_lio.pipeline import OdometryPipeline

p = load_node_params("config/ntu.yaml")
dataset = RosbagDataset(bag_dir, p["imu_topic"], p["lidar_topic"], p["min_range"])
pipeline = OdometryPipeline(dataset, p["config"], p["extrinsic"]).run()
pipeline.save_tum("traj.tum")
print(pipeline.summary())
```

## Visualization (Polyscope)

`se3_lio_pipeline --visualize` opens a live [Polyscope](https://polyscope.run)
viewer: the current scan (gravity-aligned world frame) and the trajectory, with
play/pause `[space]`, step `[N]`, center `[C]`, and screenshot `[S]`.

```bash
pip install se3-lio[viz]               # adds polyscope
se3_lio_pipeline eee_01.bag --config config/ntu.yaml \
    --max-frames 1500 --visualize
```

The viewer lives in [se3_lio/viz/polyscope_viz.py](se3_lio/viz/polyscope_viz.py)
(`PolyscopeVisualizer`, lazily importing `polyscope`). It implements the pipeline
logger interface (`log_frame`), following KISS-ICP / GenZ-ICP's Polyscope viewers.

## Visualization (Rerun)

`--rerun-save` writes a [Rerun](https://rerun.io) `.rrd` (open it in the viewer,
below): a 3D view (sliding submap, trajectory, sensor pose with TF axes,
camera-follow) plus scalar plots (linear/angular speed, compute time, CPU, RAM).
Logger: [se3_lio/viz/rerun_logger.py](se3_lio/viz/rerun_logger.py).

```bash
pip install se3-lio[rerun]             # adds rerun-sdk (pinned, see below)
se3_lio_pipeline eee_01.bag --config config/ntu.yaml --rerun-save out.rrd
```

**Version:** the `[rerun]` extra pins **`rerun-sdk==0.33.1`**, because `.rrd`
files are only guaranteed to open in a **matching viewer version** and the
logger uses version-specific APIs. Keep the viewer in lockstep — since the host
Python may be too old for a recent SDK, use a dedicated env:

```bash
conda create -n rerun033 python=3.11 -y
conda run -n rerun033 pip install rerun-sdk==0.33.1
conda run -n rerun033 rerun out.rrd    # native viewer (matches the .rrd)
```

Alternatively serve the web viewer from the same SDK (`rr.serve_web_viewer`) so
no local install has to match. When upgrading rerun, bump the pin, fix any
logger API breaks, rebuild, and **regenerate the `.rrd`** (or `rerun rrd migrate`).

### Raw binding

The `_`-prefixed names (`se3_lio_pybind._SE3LIO`, `_SE3LIOConfig`, `_State`) are
the raw binding the high-level API is built on.

```python
import numpy as np
from se3_lio import se3_lio_pybind as lio

config = lio._SE3LIOConfig()
config.downsample_resolution = 0.5
config.max_iter = 4

odom = lio._SE3LIO(config, np.eye(4))  # second arg: 4x4 LiDAR->body extrinsic

# points:      (N, 3) float64 xyz in the LiDAR frame
# point_times: (N,)   float64 per-point time offset from frame start [s]
# imu:         (M, 7) float64 rows of [t, ax, ay, az, gx, gy, gz]
state, cloud = odom._register_frame(points, point_times, imu, frame_stamp)

print(state.pose)        # 4x4 SE(3) pose; cloud: (N,3) deskewed body-frame points
print(state.vel, state.bg, state.ba, state.grav)
print(state.covariance)  # 18x18
```
