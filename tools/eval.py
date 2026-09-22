#!/usr/bin/env python3
"""ATE evaluation with the same rules as the Codabench scorer. evo does the computation.

  pairing  evo sync.associate_trajectories(t_max)
  t_max    min(GT interval, submission interval)/2 + 5 ms, capped at 50 ms   <- measured from the scorer log
  ATE     evo main_ape.ape(point_distance, align, n_to_align=-1, correct_scale=False)
          arguments as in the organizers' grand_tour_box/.../grandtour_point_relation.py.
  The scorer runs evo 1.36.5 (rslethz/grandtour-codabench:py311).

Usage: python3 tools/eval.py <est.tum> <gt.tum> [--t-max S] [--no-align]
      python3 tools/eval.py --batch <results_dir> [--csv out.csv]   # <seq>/<seq>.tum + gt.tum
"""
import argparse, csv, glob, os, sys
import numpy as np
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.main_ape import ape
from evo.tools import file_interface


def t_max_rule(t_ref, t_est):
    dt = min(np.median(np.diff(t_ref)), np.median(np.diff(t_est)))
    return min(dt / 2 + 0.005, 0.05)


def evaluate(est, gt, t_max=None, align=True):
    ref = file_interface.read_tum_trajectory_file(gt)
    e = file_interface.read_tum_trajectory_file(est)
    n_gt = ref.num_poses
    tm = t_max if t_max is not None else t_max_rule(ref.timestamps, e.timestamps)
    ref, e = sync.associate_trajectories(ref, e, tm)
    if ref.num_poses < 10:
        return {'ate_rmse': float('inf'), 'n_pair': ref.num_poses, 'n_gt': n_gt, 't_max_ms': tm * 1e3}
    s = ape(ref, e, PoseRelation.point_distance, align=align, correct_scale=False,
            n_to_align=-1, align_origin=False).stats
    return {'ate_rmse': s['rmse'], 'ate_median': s['median'], 'ate_max': s['max'],
            'n_pair': ref.num_poses, 'n_gt': n_gt, 't_max_ms': tm * 1e3}


def fmt(seq, r):
    if r['ate_rmse'] == float('inf'):
        return f'{seq:8s} pairs {r["n_pair"]}/{r["n_gt"]} too few'
    return (f'{seq:8s} ATE {r["ate_rmse"]*100:6.2f} cm  med {r["ate_median"]*100:6.2f}  max {r["ate_max"]*100:6.1f}'
            f'  pairs {r["n_pair"]}/{r["n_gt"]} ({100*r["n_pair"]/r["n_gt"]:.0f}%)  t_max {r["t_max_ms"]:.0f} ms')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('est', nargs='?')
    ap.add_argument('gt', nargs='?')
    ap.add_argument('--batch', help='results directory: each subfolder\'s gt.tum + every other *.tum that is not _imu')
    ap.add_argument('--t-max', type=float)
    ap.add_argument('--no-align', action='store_true')
    ap.add_argument('--csv')
    a = ap.parse_args()
    if a.batch:
        rows = []
        for d in sorted(glob.glob(os.path.join(a.batch, '*/'))):
            name = os.path.basename(d.rstrip('/'))
            gt = os.path.join(d, 'gt.tum')
            ests = [p for p in glob.glob(os.path.join(d, '*.tum')) if not p.endswith(('gt.tum', '_imu.tum'))]
            if len(ests) == 1 and os.path.exists(gt):
                r = evaluate(ests[0], gt, a.t_max, not a.no_align)
                rows.append({'run': name, **r})
                print(fmt(name, r))
        ok = [r['ate_rmse'] for r in rows if r['ate_rmse'] != float('inf')]
        if ok:
            print(f'{"mean":8s} ATE {np.mean(ok)*100:6.2f} cm  ({len(ok)}/{len(rows)} missions)')
        if a.csv and rows:
            with open(a.csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    elif a.est and a.gt:
        print(fmt(os.path.basename(a.est), evaluate(a.est, a.gt, a.t_max, not a.no_align)))
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
