# SE(3)-LIVOM

Multi-LiDAR-inertial odometry on SE(3), built on [url-kaist/se3-lio](https://github.com/url-kaist/se3-lio).
Entry for the COMFORT Localization Benchmark (GrandTour, IROS 2026 Data in Field Robotics workshop).

Default pipeline: Hesai XT32 + Livox Mid-360 merged in the core, STIM320 IMU, and one fisheye camera (Alphasense front_center) as a sparse-direct photometric update after the LiDAR update. `--lio-only` turns the camera off.

`--hba` adds an online HBA backend (hku-mars/HBA, GPL-2.0, vendored under `src/se3-lio/cpp/se3_lio/hba/`): the lower-layer window BAs run on a worker thread while scans arrive, and the top-layer BA plus pose-graph optimization (GTSAM) run once after the last scan. The refined trajectory is written next to the LIO one as `<seq>_hba.tum`.

## Real-time environment

All numbers are measured inside Docker with fixed CPU and memory limits, starting from the smallest budget that meets 10 Hz (every scan < 100 ms) and raising it only when a mission exceeds it.

- CPU: (to be filled)
- RAM: (to be filled)
- GPU: none

## Reproduce

(to be filled: docker build → extract → run → evaluate → submission zip)

## License

GPL-2.0, same as se3-lio.
