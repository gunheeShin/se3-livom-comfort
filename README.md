# SE(3)-LIVOM

Multi-LiDAR-inertial odometry on SE(3), built on [url-kaist/se3-lio](https://github.com/url-kaist/se3-lio).
Entry for the COMFORT Localization Benchmark (GrandTour, IROS 2026 Data in Field Robotics workshop).

Default pipeline: Hesai XT32 + Livox Mid-360 merged in the core, STIM320 IMU, and one fisheye camera (Alphasense front_center) as a sparse-direct photometric update after the LiDAR update. `--lio-only` turns the camera off.

`--hba` adds an online backend (plane bundle adjustment of hku-mars/HBA, GPL-2.0, vendored under `src/se3-lio/cpp/se3_lio/hba/`, pose graph with GTSAM). Keyframes of 25 scans accumulate while the robot moves; every 4 new keyframes a worker thread re-solves the BA over all keyframes from the LIO poses and feeds the all-pair relative poses into an incremental pose graph (iSAM2) together with per-scan odometry factors and gravity factors from stationary segments. Nothing runs after the last scan: the pose graph estimate at that moment is the output, `<seq>_hba.tum`, next to the LIO one.

## Real-time environment

All numbers are measured inside Docker with fixed limits: 8 CPUs (`--cpus=8`, pinned to the P-cores of an Intel Core i9-13900), 16 GB of RAM (`--memory=16g`), no GPU. The frontend runs on 4 threads (`--omp 4`) and the backend on 4 (`--hba-set threads=4`).

On the 12 report missions played from the bags at 1x (`--online-bag`): frontend 28–42 ms per scan (100 ms exceeded on 10 scans of CON-4 only), data-to-pose latency 51–143 ms mean, one backend round every 10 s taking 0.6–18 s, peak resident memory 5.3–16.7 GB (the longest missions, ARC-3 and CON-4, sit at the 16 GB cap; the visual map takes most of it).

## Reproduce

Data: the GrandTour mission folders `<id>_<MISSION>_release_<date>/` with their bags (`*_hesai.bag`, `*_livox.bag`, `*_stim320_imu.bag`, `*_alphasense.bag`, `*_tf_minimal.bag`). `--online-bag` reads the bags directly; the offline path needs `tools/extract.py` once per mission (see `scripts/run_se3lio.sh`). Both give the same frontend trajectory bit for bit.

```bash
bash docker/build_docker.sh                 # image comfort:ros1 (ROS Noetic, PCL, GTSAM, the se3_lio Python binding)
# one mission, played at 1x, with the backend — what submission 935484 ran:
HRUN_FROM=comfort hrun --mem 16 --cpus 8 --gpu 0 --pin p \
  bash scripts/run_se3lio.sh <seq> --lidar multi --imu-dt <dt> --tag ad-n2000 --rt --omp 4 --online-bag --hba --hba-set threads=4
# frontend only — what submission 933011 ran:
HRUN_FROM=comfort hrun --mem 16 --cpus 8 --gpu 0 --pin p \
  bash scripts/run_se3lio.sh <seq> --lidar multi --imu-dt <dt> --tag ad-n2000 --rt --omp 4 --online-bag
```

`hrun` is our job queue; it only wraps `docker run --rm --cpus=8 --memory=16g --cpuset-cpus=<P-cores>`. Without it, run the same `scripts/run_se3lio.sh` line inside such a container with `HRUN_PIN` set to the P-core list. The exact inner command, config, commit and image id of every run are written to `results/<name>/provenance/run.txt`.

Per-mission IMU time offset `--imu-dt` (seconds, STIM320 has no hardware sync; measured once per mission by cross-correlating the trajectory rotation with the gyro):

| ARC-2 | ARC-7 | CON-4 | EIG-1 | SNOW-2 | SPX-2 | ARC-3 | ARC-6 | CON-3 | EIG-2 | SNOW-3 | SPX-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -0.0073 | -0.0047 | -0.0064 | -0.0057 | -0.0055 | -0.0074 | -0.0056 | -0.0065 | -0.0060 | -0.0054 | -0.0048 | -0.0044 |

Outputs in `results/<seq>-se3lio-multi-livo-ad-n2000[-hba]-rt-online/`: `<seq>_imu.tum` (frontend, IMU frame), `<seq>.tum` (frontend, prism frame = submission format, via `tools/to_prism.py`), `<seq>_hba.tum` / `<seq>_hba_prism.tum` (backend), `timing.csv` (per-scan ms and RSS), `latency.csv`, `hba_rounds.csv`, `run.log`. Validation missions are scored against `comfort_offline/gt.tum` with `tools/eval.py` (same pairing rule as the Codabench scorer).

Submission zip: the six Test prism trajectories renamed `<seq>.tum` (10 Hz raw output, no grid, no gaps):

```bash
# 935484: backend output          933011: frontend output
for s in arc-2 arc-7 con-4 eig-1 snow-2 spx-2; do
  cp results/$s-se3lio-multi-livo-ad-n2000-hba-rt-online/${s}_hba_prism.tum tmp/$s.tum   # 933011: results/$s-...-ad-n2000-rt-online/$s.tum
done; (cd tmp && zip -j ../submission.zip *.tum)
```

Reproduction status (2026-09-21, commit with this README): the frontend trajectories of all 12 missions are bit-identical (md5) to the 933011 submission files and to earlier runs; the backend trajectories of 935484 reproduce within 0.5 cm (its rounds run on wall-clock time, so which keyframes each round covers can shift by one).

Scores (Codabench, average ATE over the six Test missions): 933011 frontend 1.66 cm, 935484 frontend + backend 1.40 cm.

## License

GPL-2.0, same as se3-lio.
