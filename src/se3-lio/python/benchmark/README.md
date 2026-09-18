# SE(3)-LIO parameter-sweep benchmark

Grid-sweeps SE(3)-LIO's parameters over one or more sequences, evaluates
rigid-aligned ATE against ground truth, and ranks the parameter combinations.
Run and scoring are decoupled: `run.py` generates trajectories, `score.py` ranks
them (so you can re-rank without re-running).

## Install

```bash
pip install se3-lio          # the odometry core (see the repo README)
```
The pure helpers (`evaluate.py`, grid expansion) use only numpy + stdlib; the
sweep runner imports the compiled `se3_lio` core. The `docker/python` image
(`se3_lio:py`) has everything and mounts data at `/ws/data`.

## Data

The benchmark does **not** ship dataset files — download each from its source and
put it where the registry expects. The paths in `datasets/*.yaml` use the
container mount `/ws/data/...`; mount your download there, or edit the paths.

| registry | dataset | sensor | source |
|---|---|---|---|
| `datasets/ntu.yaml` | NTU VIRAL | Ouster OS1 | ntu-viral-dataset.github.io |
| `datasets/ncd.yaml` | Newer College (NCD) | Ouster | ori-drs.github.io/newer-college-dataset |
| `datasets/grandtour.yaml` | GrandTour (ETH RSL) | Hesai + Livox IMU | *(add official link)* |
| `datasets/oxspires.yaml` | Oxford Spires | Hesai QT64 | dynamic.robots.ox.ac.uk/datasets/oxford-spires |
| `datasets/tartandrive.yaml` | TartanDrive | Velodyne (off-road) | github.com/castacks/tartan_drive |

> ⚠️ Each dataset has its **own licence and required citation** — cite the
> datasets you use and follow their terms. Verify/finalize the links and bibtex
> above before publishing.

Each registry entry maps a sequence key → `{bag, gt_tum, topics, base_params,
max_frames}`. `base_params` points at the repo's `config/<dataset>.yaml`.

## Run

From the repo root (inside `se3_lio:py`, data at `/ws/data`):

```bash
# 1) generate trajectories for a dataset family (or a single sequence key)
python python/benchmark/run.py   ntu --sweep python/benchmark/sweeps/basic.yaml

# 2) rank the combos vs ground truth
python python/benchmark/score.py ntu --topic basic
```
- **run.py** expands the sweep grid, runs each combo over the sequence(s), and
  writes `results/<target>/<topic>/combo_NNNN.tum` (+ a RAM/time JSON sidecar).
  Combos run in parallel; each bag is parsed once and shared to workers via fork.
- **score.py** rigid-aligns each combo against GT, ranks by mean ATE RMSE, and
  writes `scores.csv` + `best.yaml`.

`<target>` is a dataset family (`datasets/<target>.yaml` → all its keys) or a
single sequence key. Outputs land under `results/` (git-ignored).

For long sweeps, `adaptive.sh` picks the largest `--jobs` the machine fits
automatically — it starts optimistic and, on a RAM-overload abort, backs off by
one and resumes (finished combos are kept):

```bash
python/benchmark/adaptive.sh ntu --topic basic --sweep python/benchmark/sweeps/basic.yaml
```

## Layout

| path | what |
|---|---|
| `run.py` / `score.py` | sweep runner / scorer (decoupled) |
| `evaluate.py` | SE(3)-aligned ATE (numpy/stdlib only, host-importable) |
| `watchdog.py` | RAM watchdog to abort a runaway sweep |
| `adaptive.sh` | auto-jobs wrapper: backs off `--jobs` on RAM overload + resumes |
| `datasets/` | dataset registry, one yaml per family |
| `sweeps/` | grid specs — `basic` (default), `tartan`, `verify` (base-only smoke) |
| `results/` | generated trajectories + scores (git-ignored) |
