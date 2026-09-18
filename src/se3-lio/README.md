<div align="center">
    <h1>SE(3)-LIO</h1>
    <img src="https://img.shields.io/badge/-C++-blue?logo=cplusplus" />
    <img src="https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/ROS1-Noetic-blue" />
    <img src="https://img.shields.io/badge/ROS2-Humble-blue" />
    <img src="https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu&logoColor=white" />
    <img src="https://img.shields.io/badge/License-GPL--2.0-green" />
    <a href="https://pypi.org/project/se3-lio/"><img src="https://img.shields.io/pypi/v/se3-lio?logo=pypi&logoColor=white&label=PyPI" /></a>
    <br />
    <a href="https://arxiv.org/abs/2603.16118"><img src="https://img.shields.io/badge/arXiv-b33737?logo=arXiv" /></a>
    <a href="https://youtu.be/PSinH_y6MH4"><img src="https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white" /></a>
    <a href="https://se3-lio.github.io/"><img src="https://img.shields.io/badge/Project_Page-2ea44f" /></a>
    <br />
    <br />
    <p align="center">
        <a href="https://youtu.be/PSinH_y6MH4"><img src="pictures/se3-lio-demo.gif" alt="SE(3)-LIO demo (click for full video)" width="640" /></a>
        <br />
        <sub><a href="https://youtu.be/PSinH_y6MH4">▶ Click the demo for the full video</a></sub>
    </p>
    <p><strong><em>Smooth IMU propagation with jointly distributed poses on the SE(3) manifold for accurate and robust LiDAR-Inertial Odometry.</em></strong></p>
</div>

______________________________________________________________________

## :open_file_folder: Layout

SE(3)-LIO ships as ROS1 and ROS2 nodes built on a shared ROS-agnostic C++ core:

- `cpp/se3_lio/` — ROS-agnostic C++ core (state estimation, map management, pipeline)
- `ros/ros1/` — ROS1 (Noetic, catkin) node, launch, config, rviz
- `ros/ros2/` — ROS2 (Humble, ament/colcon) node, launch, config, rviz
- `config/` — dataset configs (extrinsic + topics) for the Python CLI / benchmark
- `docker/ros1/`, `docker/ros2/` — Docker setup for building and running each

______________________________________________________________________

## :package: Build & Run

All build and run steps run **inside Docker** — on the host you only build the
image and drop into the container.

<details>
<summary><b>ROS 1 (Noetic)</b></summary>

### Build

```bash
# host
bash docker/ros1/build_docker.sh     # builds se3_lio:ros1
bash docker/ros1/run_docker.sh       # drops into /ws inside the container

# inside the container
cd /ws
catkin build se3_lio                 # default ouster; -DLIDAR_TYPE=livox|hesai to switch
source devel/setup.bash
```

The LiDAR input type is a **build option** — pass
`--cmake-args -DLIDAR_TYPE=ouster|hesai|livox` (default `ouster`). No source edits
needed; `ouster`/`hesai` use `PointCloud2`, `livox` uses Livox `CustomMsg`.

### Run

```bash
roslaunch se3_lio run_se3lio_ntu.launch use_sim_time:=true   # NTU VIRAL
roslaunch se3_lio run_se3lio_ncd.launch use_sim_time:=true   # Newer College (Multi-Cam)
rosbag play --clock <data.bag>
```

`use_sim_time:=true` makes the node follow the bag clock (`rosbag play --clock`
publishes `/clock`). For live sensors, omit it — the default is `false`.
Configs live in [ros/ros1/config/ntu.yaml](ros/ros1/config/ntu.yaml)
and [ros/ros1/config/ncd.yaml](ros/ros1/config/ncd.yaml);
swap datasets by editing the `<rosparam … file=…/>` line in the launch file.
The launch opens RViz automatically — it needs a display: launch via
`run_docker.sh` and run `xhost +local:root` on the host.

</details>

<details>
<summary><b>ROS 2 (Humble)</b></summary>

### Build

```bash
# host
bash docker/ros2/build_docker.sh     # builds se3_lio:ros2
bash docker/ros2/run_docker.sh       # drops into /ws inside the container

# inside the container
cd /ws
colcon build --symlink-install --packages-select se3_lio   # default livox
source install/setup.bash
```

The LiDAR input type is a **build option** — pass
`--cmake-args -DLIDAR_TYPE=ouster|livox` (default `livox`). No source edits needed;
`ouster` uses `PointCloud2`, `livox` uses Livox `CustomMsg`.

### Run

```bash
ros2 launch se3_lio run.launch.py    # add rviz:=true for visualization
ros2 bag play <rosbag2_dir>
```

Config lives in [ros/ros2/config/params.yaml](ros/ros2/config/params.yaml)
(edit `imu_topic` / `lidar_topic` / extrinsics for your sensor rig).

`rviz:=true` opens RViz (registered cloud + trajectory). RViz needs a display:
launch via `run_docker.sh` and run `xhost +local:root` on the host.

</details>

<details>
<summary><b>Python bindings (pybind11)</b></summary>

The core is exposed to Python via pybind11 ([python/](python/)). Linux x86_64
wheels are on PyPI, so most users can just:

```bash
pip install se3-lio          # add [viz] for the live Polyscope viewer
```

### Build from source

The C++ core needs OpenMP (libgomp); Eigen and Sophus are fetched and compiled
in. Build **inside the ROS2 Docker image**, where OpenMP is available.

```bash
# inside the se3_lio:ros2 container
pip install pybind11 scikit-build-core pydantic
pip install --no-build-isolation ./python/      # self-contained wheel ($ORIGIN RPATH)
pytest python/tests/
```

### Use

```python
from se3_lio import SE3LIO, SE3LIOConfig, load_node_params

p = load_node_params("config/ntu.yaml")   # same mapping as the node
odom = SE3LIO(p["config"], p["extrinsic"])

# points (N,3) · point_times (N,) offsets [s] · imu (M,7) [t,ax,ay,az,gx,gy,gz]
state = odom.register_frame(points, point_times, imu, frame_stamp)
print(state.pose, state.vel, state.covariance)
```

`register_frame` reproduces the node's per-frame data path (build measurement →
apply extrinsic → sort by timestamp → `estimatePose`).

### CLI pipeline

`se3_lio_pipeline` runs odometry over a rosbag and writes a TUM trajectory — the
offline Python equivalent of the node. Input is auto-detected: a rosbag2
directory → ROS2/Livox, a `*.bag` file → ROS1/Ouster.

```bash
# ROS2 / Livox
se3_lio_pipeline <rosbag_dir> --config my_rig.yaml --imu-topic /livox/imu --lidar-topic /livox/lidar --max-frames 300
# ROS1 / Ouster (e.g. NTU VIRAL)
se3_lio_pipeline eee_01.bag --config config/ntu.yaml --max-frames 1500
# ROS1 / Ouster (e.g. Newer College Multi-Cam)
se3_lio_pipeline MathsHard_MC.bag --config config/ncd.yaml --max-frames 1500
```

Add `--visualize` for a live [Polyscope](https://polyscope.run) viewer (current
scan + trajectory, play/pause/step) — needs `pip install se3-lio[viz]`.

See [python/README.md](python/README.md) for the full API and supported inputs.

</details>

______________________________________________________________________

## :gear: Configuration

The same parameter groups apply to both ROS1 and ROS2 (in their respective
`config/` yaml):

| Group | Key | Meaning |
|-------|-----|---------|
| topics | `imu_topic`, `lidar_topic` | input topic names |
| `sensors/imu` | `acc_cov`, `gyr_cov`, `b_acc_cov`, `b_gyr_cov` | IMU noise / bias covariances |
| `sensors/lidar` | `min_range`, `range_cov`, `angle_cov` | LiDAR range gate and measurement noise |
| `sensors/lidar` | `point_filter_num` | keep every Nth raw point (1 = keep all) |
| `sensors` | `t_exts`, `q_exts` | LiDAR→IMU extrinsic (translation, quaternion `w,x,y,z`) |
| `downsample` | `resolution` | input downsample voxel size (m) |
| | `max_iter` | max iterations per update |
| `voxel_map` | `resolution`, `max_layer`, `layer_size`, `max_point_size`, `plane_threshold` | voxel map structure and plane fitting |

______________________________________________________________________

## :construction: Roadmap

- [x] ROS2 (Humble) support
- [x] Python bindings (pybind11) + `se3_lio_pipeline` CLI (ROS2/Livox, ROS1/Ouster); published on PyPI (Linux wheels)
- [ ] More dataset configs and example launches
- [ ] Continuous integration (build checks)

______________________________________________________________________

## :scroll: Citation

If you use SE(3)-LIO in your academic work, please cite our
[paper](https://arxiv.org/abs/2603.16118):

```bibtex
@inproceedings{shin2026icra,
  author    = {Shin, Gunhee and Lee, Seungjae and Kong, Jei and Seo, Youngwoo and Myung, Hyun},
  title     = {SE(3)-LIO: Smooth IMU Propagation With Jointly Distributed Poses on SE(3) Manifold for Accurate and Robust LiDAR-Inertial Odometry},
  booktitle = {Proc. of the IEEE Int. Conf. on Robotics and Automation (ICRA)},
  year      = {2026},
}
```

______________________________________________________________________

## :pray: Acknowledgements

- The voxel map ([cpp/se3_lio/core/temporary/voxel_map_util.{h,cpp}](cpp/se3_lio/core/temporary/voxel_map_util.h))
  is adapted from [HKU-MARS VoxelMap](https://github.com/hku-mars/VoxelMap) (GPL-2.0).
- Thanks to [KISS-ICP](https://github.com/PRBonn/kiss-icp) and
  [GenZ-ICP](https://github.com/cocel-postech/genz-icp), which we referenced for
  the project layout and code structure.

______________________________________________________________________

## License

This project is released under **GPL-2.0**, inherited from the bundled
[HKU-MARS VoxelMap](https://github.com/hku-mars/VoxelMap) component. See [LICENSE](LICENSE).
