#!/bin/bash
# bash scripts/run.sh <seq> frontend   frontend only (submission 933011)
# bash scripts/run.sh <seq>            frontend + backend (submission 935484, the leaderboard entry)
# <seq>: mission name in lower case (arc-2). DATA must point at the folder with the mission folders.
set -e
cd "$(dirname "$0")/.."
ARGS="--lidar multi --tag ad-n2000 --rt --omp 4 --online-bag"
[ "${2:-}" = frontend ] || ARGS="$ARGS --backend --backend-set threads=4"
bash scripts/run_se3lio.sh "$1" $ARGS
