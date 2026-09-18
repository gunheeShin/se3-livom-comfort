"""Benchmark runner: `python3 python/benchmark/run.py <target>` (from repo root).

Expands a grid sweep and, for each combo, runs SE(3)-LIO over one or more bags,
evaluates rigid-aligned ATE against ground truth, and ranks combos. With a
single bag the score is that bag's ATE RMSE; with a suite (the sweep yaml lists
`datasets:`) the score is the mean of per-bag RMSE, so the winner generalizes
across bags. Writes scores.csv + best.yaml. The compiled binding is imported
lazily inside the functions that actually execute a sweep, so the pure helpers
(load_registry / expand_grid) stay host-importable.
"""

import argparse
import copy
import json
import os
import resource
import tempfile
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

# Force the core's voxel-map OMP loops to a single thread so each combo is
# deterministic (run-to-run identical ATE). Set before the binding is imported
# and inherited by the forked Pool workers. Requires the core built with the
# SE3_LIO_OMP_THREADS env hook (voxel_map_util.cpp).
os.environ.setdefault("SE3_LIO_OMP_THREADS", "1")

import yaml

# Make the sibling `benchmark` package importable when run as a plain script.
# Appended (not prepended) so the pip-installed se3_lio with its compiled
# binding still wins over the uncompiled source tree under python/se3_lio/.
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


# --- dataset registry (key -> bag/base_params/GT/topics) ------------------
_OVERRIDE_KEYS = (
    "imu_topic", "lidar_topic", "min_range", "point_filter_num", "max_frames",
    "input_type", "bag", "gt_tum",
)


def load_registry(path):
    """Load the dataset registry into {dataset_key: entry}.

    `path` is a directory of per-dataset-family yamls (one file per family, each
    a dict of sequence_key -> entry); they are merged into one registry. A single
    yaml file is also accepted.
    """
    p = Path(path)
    files = sorted(p.glob("*.yaml")) if p.is_dir() else [p]
    registry = {}
    for f in files:
        with open(f) as fh:
            registry.update(yaml.safe_load(fh) or {})
    return registry


def resolve_entry(entry):
    """Merge registry fields onto the loaded base node params (binding required).

    Loads entry["base_params"] via load_node_params and overrides topics/range
    with any registry values, carrying bag/gt_tum/input_type/max_frames through.
    """
    from se3_lio.config import load_node_params

    p = load_node_params(entry["base_params"])
    resolved = {
        "config": p["config"],
        "extrinsic": p["extrinsic"],
        "imu_topic": p["imu_topic"],
        "lidar_topic": p["lidar_topic"],
        "min_range": p["min_range"],
        "point_filter_num": p["point_filter_num"],
        "base_params": entry["base_params"],
        "input_type": "ros1-ouster",
        "max_frames": None,
        "bag": None,
        "gt_tum": None,
    }
    for k in _OVERRIDE_KEYS:
        if k in entry and entry[k] is not None:
            resolved[k] = entry[k]
    return resolved


# --- grid search ----------------------------------------------------------
def expand_grid(spec):
    """Expand a sweep spec {"grid": {key: [vals]}} -> list of override dicts."""
    grid = spec.get("grid", {})
    keys = list(grid)
    if not keys:
        return [{}]
    return [dict(zip(keys, combo)) for combo in product(*(grid[k] for k in keys))]


# Frames + paths for the bag currently being swept. Populated in the parent
# before the Pool forks; workers inherit it via fork (copy-on-write).
_bag = {}


def _apply_overrides(raw, overrides):
    """Apply dotted-key overrides onto a (deep-copied) params dict in place."""
    merged = copy.deepcopy(raw)
    node = merged.get("/**", merged)
    p = node.get("ros__parameters", node)
    for dotted, value in overrides.items():
        cur = p
        keys = dotted.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    return merged


def _config_from_overrides(base_params, overrides):
    """Build an SE3LIOConfig (+extrinsic etc.) with dotted overrides applied.

    Writes the merged params dict to a temp yaml and reuses load_node_params so
    the node's exact key mapping is preserved.
    """
    from se3_lio.config import load_node_params

    with open(base_params) as f:
        raw = yaml.safe_load(f)
    merged = _apply_overrides(raw, overrides)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(merged, tf)
        tmp = tf.name
    try:
        return load_node_params(tmp)
    finally:
        os.unlink(tmp)


def _load_frames(entry, stream=False):
    """Resolve params + build the dataset (+ resolved params).

    Default: materialize the frame list once so Pool workers COW-share it (bag
    parsed once -- fast for a many-combo sweep). With ``stream=True`` return the
    streaming dataset instead; each combo re-reads the bag, so RAM stays bounded
    on huge bags (e.g. boeun) at the cost of re-parsing per combo."""
    from se3_lio.datasets import RosbagDataset, Ros1BagDataset

    params = resolve_entry(entry)
    ds_cls = Ros1BagDataset if params["input_type"] == "ros1-ouster" else RosbagDataset
    ds = ds_cls(params["bag"], params["imu_topic"], params["lidar_topic"],
                params["min_range"], params["max_frames"], params["point_filter_num"])

    # A streaming dataset re-reads the bag on each __iter__, so passing the object
    # itself lets every combo iterate it without holding all frames in memory.
    return (ds if stream else list(ds)), params


def _run_combo(args):
    """Run one override combo over the fork-inherited bag frames; save its TUM.

    No evaluation here — scoring is decoupled (see benchmark/score.py), so a run
    only produces trajectories.
    """
    from se3_lio.pipeline import OdometryPipeline

    combo_id, overrides = args
    params = _config_from_overrides(_bag["base_params"], overrides)
    pipe = OdometryPipeline(_bag["frames"], params["config"], _bag["extrinsic"])
    t0 = time.monotonic()
    pipe.run(progress=False)
    dt = time.monotonic() - t0
    tum = Path(_bag["out_dir"]) / f"combo_{combo_id:04d}.tum"
    pipe.save_tum(tum)
    pipe.save_cov(tum.with_suffix(".cov.npy"))
    # Persist RAM/time next to the TUM, the same way — so they survive an
    # interrupted run and need no end-of-run aggregation. Peak RSS (KB on Linux)
    # includes the COW-shared frames (constant across combos), so compare combos
    # *relatively*: the spread reflects param-driven voxel-map growth.
    peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    with open(tum.with_suffix(".json"), "w") as f:
        json.dump({"peak_ram_mb": peak_mb, "time_s": round(dt, 1)}, f)
    return combo_id


def _warmup():
    """Settle the core's FP regime once in THIS process (a forked worker, or the
    main process when serial), running the base config over the shared frames.
    The core's first run reshapes the heap; later runs stay stable."""
    from se3_lio.pipeline import OdometryPipeline

    OdometryPipeline(_bag["frames"], _bag["config"], _bag["extrinsic"]).run(progress=False)


def _run_bag(entry, bag_out_dir, tasks, jobs, stream=False):
    """Run every combo over one bag's frames, saving combo_NNNN.tum (no scoring).

    Default: the bag is parsed once in the parent and stashed in _bag; the Pool
    workers fork and inherit the frames (Linux copy-on-write), so no combo
    re-parses the bag. With ``stream=True`` _bag holds the streaming dataset
    instead (bounded RAM for huge bags), so each combo re-reads the bag.
    """
    bag_out_dir.mkdir(parents=True, exist_ok=True)
    # Resume: skip combos already finished. The .json sidecar is written last
    # (after the TUM), so its presence means the combo completed -- letting an
    # OOM-aborted run continue at fewer --jobs instead of restarting from zero.
    todo = [t for t in tasks
            if not (bag_out_dir / f"combo_{t[0]:04d}.json").exists()]
    if not todo:
        return
    jobs = min(jobs, len(todo))  # don't spawn idle warm-up workers for done combos
    frames, params = _load_frames(entry, stream)
    _bag.update(frames=frames, config=params["config"], extrinsic=params["extrinsic"],
                base_params=params["base_params"], out_dir=str(bag_out_dir))
    # Each combo runs in a forked worker (frames COW-shared from the parent, so no
    # combo re-parses the bag). maxtasksperchild recycles a worker after a few
    # combos: the voxel map's memory is reclaimed on respawn instead of piling up
    # across combos in one long-lived process (which OOMs on large maps). Warm-up
    # runs in each (re)spawned worker, never the parent -- the parent holds no
    # OpenMP threads at fork time, or the children would deadlock.
    with Pool(jobs, initializer=_warmup, maxtasksperchild=4) as pool:
        pool.map(_run_combo, todo)


def _resolve_ds_keys(target, config, spec=None):
    """Sequences for a target: a family (datasets/<target>.yaml -> all its keys)
    or a single sequence key. An explicit `datasets:` in the sweep spec wins."""
    if spec and spec.get("datasets"):
        return spec["datasets"]
    family = Path(config) / f"{target}.yaml"
    if Path(config).is_dir() and family.exists():
        with open(family) as f:
            return list(yaml.safe_load(f) or {})
    return [target]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchmark.run")
    ap.add_argument("target", help="dataset key (single bag) or sweep/output name (suite)")
    ap.add_argument("--config", default="python/benchmark/datasets",
                    help="dataset registry dir (per-family yamls) or a single yaml")
    ap.add_argument("--sweep", default="python/benchmark/sweeps/basic.yaml",
                    help="sweep yaml (in python/benchmark/sweeps/); the topic defaults to its stem. "
                         "basic.yaml = default grid; verify.yaml = empty grid (base params only).")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="parallel workers")
    ap.add_argument("--stream", action="store_true",
                    help="stream each bag instead of materializing all frames "
                         "(bounded RAM for huge bags; re-reads the bag per combo)")
    ap.add_argument("--out", default=None,
                    help="output dir override (default results/<target>/<topic>/)")
    ap.add_argument("--topic", default=None,
                    help="test-topic subdir: results/<target>/<topic>/ (default: the sweep yaml stem)")
    args = ap.parse_args(argv)

    registry = load_registry(args.config)

    def _out_dir(topic):
        d = Path(args.out or f"python/benchmark/results/{args.target}/{topic}")
        d.mkdir(parents=True, exist_ok=True)
        return d

    sweep_path = args.sweep
    with open(sweep_path) as f:
        spec = yaml.safe_load(f)
    combos = expand_grid(spec)
    tasks = list(enumerate(combos))

    ds_keys = _resolve_ds_keys(args.target, args.config, spec)
    for k in ds_keys:
        if k not in registry:
            ap.error(f"dataset '{k}' not in {args.config}")

    # Cap workers at the combo count: each Pool worker runs a full-bag warm-up on
    # startup, so jobs > combos just spawns wasted warm-ups (30 cores for 3 combos).
    jobs = max(1, min(args.jobs, len(tasks)))
    out_dir = _out_dir(args.topic or Path(sweep_path).stem)
    print(f"{len(combos)} combos x {len(ds_keys)} bag(s), {jobs} workers -> {out_dir}")

    # Write run_meta up front so the topic is self-describing even if the run is
    # interrupted: combo_*.tum (trajectory) + combo_*.json (peak_ram_mb, time_s)
    # accumulate per-combo as they finish, and score reads them later.
    with open(out_dir / "run_meta.yaml", "w") as f:
        yaml.safe_dump(
            {"target": args.target,
             "base_params": registry[ds_keys[0]]["base_params"],
             "ds_keys": ds_keys,
             "gt_tum": {k: registry[k]["gt_tum"] for k in ds_keys},
             "sweep": spec},
            f, sort_keys=False,
        )

    # Generate trajectories only — bags run sequentially, combos in parallel.
    for ds_key in ds_keys:
        _run_bag(registry[ds_key], out_dir / ds_key, tasks, jobs, args.stream)
        print(f"  {ds_key}: done")

    print(f"done -> {out_dir}\n"
          f"score with: python -m benchmark.score {args.target} --topic {out_dir.name}")


if __name__ == "__main__":
    main()
