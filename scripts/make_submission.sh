#!/bin/bash
# results/<seq>-<tag>/<seq>_imu.tum (Test 6) → 제출 zip. 러너 원 출력(10 Hz)을 프리즘 위치(Δl 포함, tools/to_prism.py)로 옮겨 그대로 담는다.
# 사용: bash scripts/make_submission.sh <tag> <out.zip>
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
