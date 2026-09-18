# SE(3)-LIVOM

Multi-LiDAR-inertial odometry on SE(3), built on [url-kaist/se3-lio](https://github.com/url-kaist/se3-lio).
Entry for the COMFORT Localization Benchmark (GrandTour, IROS 2026 Data in Field Robotics workshop).

Current release: LiDAR-inertial path only (Hesai XT32 + Livox Mid-360 merged in the core, STIM320 IMU). The visual path is under development.

## Real-time environment

All numbers are measured inside Docker with fixed CPU and memory limits, starting from the smallest budget that meets 10 Hz (every scan < 100 ms) and raising it only when a mission exceeds it.

- CPU: (to be filled)
- RAM: (to be filled)
- GPU: none

## Reproduce

(to be filled: docker build → extract → run → evaluate → submission zip)

## License

GPL-2.0, same as se3-lio.
