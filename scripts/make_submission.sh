#!/bin/bash
# results/<seq>-<tag>/<seq>_imu.tum (6 Test missions) -> submission zip. The raw runner output (10 Hz) is moved to the prism position (with dl, tools/to_prism.py) and packed as is.
# Usage: bash scripts/make_submission.sh <tag> <out.zip>
set -euo pipefail
WS="$(realpath "$(dirname "$0")/..")"
DATA=/media/gunhee/gun_T7_17/Research/LIO/PublichDataset/grandtour
TAG="$1"; ZIP="$(realpath -m "$2")"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
for SEQ in arc-2 arc-7 con-4 eig-1 snow-2 spx-2; do
  MDIR=$(ls -d "$DATA"/*_"${SEQ^^}"_release_* | head -1)
  python3 "$WS/tools/to_prism.py" "$MDIR/comfort_offline/calib.json" "$WS/results/$SEQ-$TAG/${SEQ}_imu.tum" "$TMP/$SEQ.tum"
done
rm -f "$ZIP"; (cd "$TMP" && zip -q -j "$ZIP" *.tum)
unzip -l "$ZIP"
