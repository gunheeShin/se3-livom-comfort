# se3_lio Python package

Python binding of the SE(3)-LIO C++ core (`../cpp/se3_lio`: synchronizer, core, visual, pipeline, backend) plus the
COMFORT dataset readers and the odometry loop that `tools/se3lio_run.py` drives.

## Build (inside the Docker image `comfort:ros1`)

```bash
pip3 install --no-build-isolation -e /ws/src/se3-lio/python
```

`scripts/run_se3lio.sh` does this automatically when `import se3_lio.pybind.se3_lio_pybind` fails.

## Layout

| path | role |
|---|---|
| `se3_lio/pybind/se3_lio_pybind.cpp` | binding: `_SE3LIO` (register_frame / register_multi), `_SE3LIOConfig`, `_BackendParams`, `_OnlineBackend` |
| `se3_lio/config.py` | `SE3LIOConfig` (pydantic) and `load_node_params()` for `config/comfort.yaml` |
| `se3_lio/se3_lio.py` | `SE3LIO`: thin wrapper over the binding |
| `se3_lio/pipeline.py` | `OdometryPipeline`: frames in, trajectory + timing.csv out, feeds the online backend |
| `se3_lio/online_sync.py` | IMU / LiDAR / image epoch synchronizer (frame rule shared by both readers) |
| `se3_lio/datasets/comfort_bag.py` | reads the mission bags directly at 1x (`--online-bag`) |
| `se3_lio/datasets/comfort_offline.py` | reads the `comfort_offline/` extract of `tools/extract.py` |
| `tests/` | smoke test of the binding and the multi-LiDAR merge: `pip3 install pytest && python3 -m pytest tests` after the install above, in the same container |
