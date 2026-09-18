"""Score a finished sweep: evaluate the saved combo TUMs against GT, rank by
mean ATE, and write scores.csv + best.yaml.

Decoupled from run.py — a run only generates trajectories; scoring reads
`<topic>/run_meta.yaml` (written by run.py) plus the per-combo TUMs, so it needs
no LIO binding (pure numpy/stdlib) and can re-rank without re-running (e.g. after
changing GT or the ATE settings).

    python -m benchmark.score <target> [--topic sweep]
"""

import argparse
import csv
import json
from pathlib import Path

# Sibling-package import when run as a plain script (see run.py).
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml

from benchmark import evaluate as ev
from benchmark.run import expand_grid


def _write_scores(results, override_keys, ds_keys, path):
    # ATE columns first, then resource columns (peak RAM / time) so best can be
    # chosen weighing accuracy against RAM.
    fields = (["combo_id"] + override_keys + [f"rmse_{k}" for k in ds_keys]
              + ["mean_rmse", "peak_ram_mb", "time_s"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})


def score_topic(topic_dir, max_dt=0.02):
    """Rank a topic's combo TUMs vs GT -> writes scores.csv + best.yaml, returns best."""
    topic_dir = Path(topic_dir)
    meta = yaml.safe_load((topic_dir / "run_meta.yaml").read_text())
    spec = meta["sweep"]
    combos = expand_grid(spec)
    override_keys = sorted(spec.get("grid", {}).keys())
    ds_keys = meta["ds_keys"]
    gt = meta["gt_tum"]

    results = []
    for cid, overrides in enumerate(combos):
        row = {"combo_id": cid}
        row.update(overrides)
        rmses, rams, times = [], [], []
        for ds_key in ds_keys:
            tum = topic_dir / ds_key / f"combo_{cid:04d}.tum"
            rmse = (ev.evaluate(str(tum), gt[ds_key], max_dt=max_dt)["ate_rmse"]
                    if tum.exists() else float("inf"))
            row[f"rmse_{ds_key}"] = rmse
            rmses.append(rmse)
            # RAM/time sidecar written next to each TUM by run.py.
            sidecar = tum.with_suffix(".json")
            if sidecar.exists():
                sidecar_meta = json.loads(sidecar.read_text())
                rams.append(sidecar_meta.get("peak_ram_mb"))
                times.append(sidecar_meta.get("time_s"))
        row["mean_rmse"] = sum(rmses) / len(rmses)  # any inf propagates -> combo ranks last
        # peak RAM = max across bags; time = total across bags.
        rams = [x for x in rams if x is not None]
        times = [x for x in times if x is not None]
        row["peak_ram_mb"] = max(rams) if rams else ""
        row["time_s"] = round(sum(times), 1) if times else ""
        results.append(row)

    results.sort(key=lambda r: (r["mean_rmse"], r["combo_id"]))
    _write_scores(results, override_keys, ds_keys, topic_dir / "scores.csv")

    best = results[0]
    best_overrides = {k: best[k] for k in override_keys}
    with open(topic_dir / "best.yaml", "w") as f:
        yaml.safe_dump(
            {"base_params": meta["base_params"],
             "datasets": ds_keys,
             "overrides": best_overrides,
             "mean_rmse": best["mean_rmse"],
             "per_bag_rmse": {k: best[f"rmse_{k}"] for k in ds_keys}},
            f, sort_keys=False,
        )
    return best, best_overrides


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchmark.score")
    ap.add_argument("target", help="dataset/family name -> results/<target>/<topic>/")
    ap.add_argument("--topic", default="sweep", help="topic subdir (default: sweep)")
    ap.add_argument("--dir", default=None, help="topic dir override (else results/<target>/<topic>)")
    ap.add_argument("--max-dt", type=float, default=0.02, help="GT association window (s)")
    args = ap.parse_args(argv)
    topic_dir = args.dir or f"python/benchmark/results/{args.target}/{args.topic}"
    best, best_overrides = score_topic(topic_dir, max_dt=args.max_dt)
    print(f"scored {topic_dir} -> scores.csv + best.yaml")
    print(f"best mean_rmse={best['mean_rmse']:.4f} -> {best_overrides}")


if __name__ == "__main__":
    main()
