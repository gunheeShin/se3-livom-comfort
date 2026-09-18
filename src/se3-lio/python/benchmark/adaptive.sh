#!/usr/bin/env bash
# Adaptive-jobs benchmark runner. Start optimistic (--start-jobs), and on a RAM
# overload abort (watchdog exit 137) back off --jobs by one and resume -- run.py
# skips combos whose .json sidecar already exists, so finished work is kept. This
# auto-settles at the largest --jobs the machine fits, with no hand tuning and no
# cross-run state (each invocation is self-contained).
#
#   python/benchmark/adaptive.sh <target> --topic <t> \
#       [--sweep <file>] [--start-jobs N] [--max-ram-pct P]
#
# Run from the repo root (inside the se3_lio:py container).
set -u

T=${1:?usage: adaptive.sh <target> --topic <t> [--sweep <f>] [--start-jobs N] [--max-ram-pct P]}
shift
TOPIC=demo; SWEEP=""; JOBS=4; PCT=88
while [ $# -gt 0 ]; do
  case "$1" in
    --topic)       TOPIC=$2; shift 2;;
    --sweep)       SWEEP="--sweep $2"; shift 2;;
    --start-jobs)  JOBS=$2; shift 2;;
    --max-ram-pct) PCT=$2; shift 2;;
    *) echo "adaptive.sh: unknown arg '$1'" >&2; exit 2;;
  esac
done

OUT=python/benchmark/results/$T/$TOPIC
rm -rf "$OUT"   # fresh run; combos then accumulate as resumable checkpoints

while :; do
  echo "[adaptive] $T/$TOPIC -- attempt at --jobs $JOBS (RAM cap ${PCT}%)"
  python3 python/benchmark/watchdog.py --max-ram-pct "$PCT" --interval 10 -- \
    python3 python/benchmark/run.py "$T" --topic "$TOPIC" $SWEEP --jobs "$JOBS"
  rc=$?
  if [ "$rc" -eq 137 ] && [ "$JOBS" -gt 1 ]; then
    JOBS=$((JOBS - 1))
    echo "[adaptive] RAM overload -> back off to --jobs $JOBS, resuming finished combos"
    continue
  fi
  break
done

python3 python/benchmark/score.py "$T" --topic "$TOPIC"
