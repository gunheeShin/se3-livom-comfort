#!/bin/bash
# 한 미션 SE(3)-LIO 실행: extract 확인 → (바인딩 없으면 빌드) → se3lio_run → prism 변환 → 평가.
# 호스트에서: HRUN_FROM=se3lio-multi hrun --mem 16 --cpus 8 --gpu 0 --pin p bash scripts/run_se3lio.sh <seq> [--tag NAME] [--lidar merged] [--imu-dt S] [--config PATH] [--set KEY=VAL ...]
#   --set KEY=VAL: SE3LIOConfig 필드 덮어쓰기(max_iter=10). 여러 번 줄 수 있다
#   결과 results/<seq>-se3lio[-<tag>]/{<seq>_imu.tum,<seq>.tum,gt.tum,run.log,provenance/}
#   --lidar merged: comfort_offline/lidar_merged/ (없으면 scripts/run.sh 로 먼저 만든다)
#   --lidar multi:  comfort_offline/lidar/ + livox/ 를 따로 읽어 se3-lio 코어가 병합(중복 제거 없음). livox/ 없으면 extract --livox-only
#   --cam:          이미지 스탬프가 에폭 경계(pose 스탬프 = 이미지 시각, 사진은 안 씀). 결과 이름에 -cam
#   --livo:         --cam + photometric 갱신(visual_en=1). 결과 이름에 -livo
#   --cams A,B:     카메라 목록(기본 front_center). 여럿이면 이름에 -<N>cam. cam_<name>/ 없으면 extract --cams-only
#   --img-dt S:     이미지 시각 오프셋(초, se3lio_run --img-time-offset)
#   --rt:           공개(계측) 프로필 — OMP 8 스레드, 결과 이름에 -rt. hrun --cpus all 로 혼자 돌린다(ms/scan 은 옆 잡에 흔들린다)
#   기본(내부) 프로필은 OMP 4 스레드 — hrun --cpus 8 로 동시 3개. 어느 쪽이든 timing.csv 는 남는다
#   --dump:         디스큐 스캔을 results/<name>/scans/00000.pcd… 로 저장(궤적 행과 1:1, HBA 입력, 미션당 3~8 GB)
#   --stage-timing: 코어 3구간 시간(predict·update·map ms)을 run.log 에 TIMING 줄로 남긴다(SE3LIO_TIMING=1). update = LiDAR + 카메라 갱신, map = 복셀 + visual 맵
#   --viz:          시각화 덤프 results/<name>/viz.npz. 호스트에서 python3 tools/viz_rrd.py results/<name>/viz.npz → viz.rrd (rerun-sdk·뷰어 0.33.1)
set -euo pipefail
WS="$(realpath "$(dirname "$0")/..")"
DATA=/media/gunhee/gun_T7_17/Research/LIO/PublichDataset/grandtour
SEQ="$1"; shift; ARGS="$*"
TAG=""; IMUDT=""; LIDAR=""; CFG=/ws/src/se3-lio/config/comfort.yaml; SET=""; CAM=""; LIVO=""; IMGDT=""; CAMS=front_center; DUMP=""; RT=""; VIZ=""
while [ $# -gt 0 ]; do case "$1" in --tag) TAG="-$2"; shift;; --imu-dt) IMUDT="$2"; shift;; --lidar) LIDAR="$2"; shift;; --config) CFG="$2"; shift;; --set) SET="$SET --set $2"; shift;; --cam) CAM=1;; --cams) CAMS="$2"; shift;; --img-dt) IMGDT="$2"; shift;; --dump) DUMP=1;; --rt) RT=1;; --viz) VIZ=1;; --stage-timing) export SE3LIO_TIMING=1;; --livo) CAM=1; LIVO=1; SET="$SET --set visual_en=1";; esac; shift; done

MDIR=$(ls -d "$DATA"/*_"${SEQ^^}"_release_* | head -1)
OFF="$MDIR/comfort_offline"
LIDAR_DIR="$OFF/lidar${LIDAR:+_$LIDAR}"; LIVOX=""
[ "$LIDAR" = multi ] && { LIDAR_DIR="$OFF/lidar"; LIVOX="--livox-dir $OFF/livox"; }
MODE=""; [ -z "$CAM" ] || MODE="-cam"; [ -z "$LIVO" ] || MODE="-livo"
NCAM=$(tr , "\n" <<<"$CAMS" | wc -l); [ "$NCAM" -le 1 ] || MODE="$MODE-${NCAM}cam"
NAME="$SEQ-se3lio${LIDAR:+-$LIDAR}$MODE$TAG${RT:+-rt}"
export OMP_NUM_THREADS=$([ -n "$RT" ] && echo 8 || echo 4)
[ -z "$CAM" ] || CAM="--cams $CAMS${IMGDT:+ --img-time-offset $IMGDT}"
OUT="$WS/results/$NAME"
mkdir -p "$OUT"
SCAN_DUMP=""; [ -z "$DUMP" ] || { rm -rf "$OUT/scans"; SCAN_DUMP="--dump /ws/results/$NAME/scans"; }

PCORES=$(cat /sys/devices/cpu_core/cpus)
[ "${HRUN_PIN:-}" = "$PCORES" ] || { echo "[$SEQ] hrun --pin p 로만 돈다 (HRUN_PIN='${HRUN_PIN:-}', P코어=$PCORES)"; exit 2; }
[ -z "$VIZ" ] || VIZ="--viz /ws/results/$NAME/viz.npz"
PROV="$OUT/provenance"; mkdir -p "$PROV"
git -C "$WS" rev-parse HEAD > "$PROV/commit"
{ git -C "$WS" diff --binary HEAD
  git -C "$WS" ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do git -C "$WS" diff --binary --no-index /dev/null "$f" || true; done
} > "$PROV/diff.patch"
{ echo "date: $(date -Is)"; echo "cmd: $0 $SEQ $ARGS"; echo "HRUN_PIN=$HRUN_PIN HRUN_CPUS=${HRUN_CPUS:-} HRUN_MEM=${HRUN_MEM:-}"
  echo "image: $(docker image inspect --format '{{.Id}}' "${IMAGE:-comfort:ros1}")"; echo "lidar_dir: $LIDAR_DIR"; echo "config: $CFG"; echo "set: $SET"; echo "OMP_NUM_THREADS=$OMP_NUM_THREADS rt=${RT:-0}"; } > "$PROV/run.txt"

[ -f "$OFF/manifest.json" ] || { echo "[$SEQ] extract"; bash "$WS/docker/run_docker.sh" python3 /ws/tools/extract.py "$MDIR" --cam none; }
[ -z "$LIDAR" ] || [ "$LIDAR" = multi ] || [ -f "$LIDAR_DIR.done" ] || { echo "[$SEQ] $LIDAR_DIR 없음 — scripts/run.sh --lidar $LIDAR 로 먼저 만든다"; exit 1; }
[ -z "$CAM" ] || [ -d "$OFF/cam" ] || { echo "[$SEQ] $OFF/cam 없음 — extract 를 --cam front_center 로"; exit 1; }
[ -z "$CAM" ] || python3 -c "import json,sys; c=json.load(open('$OFF/calib.json')); sys.exit(0 if all(n in c.get('cams',{}) for n in '$CAMS'.split(',')) else 1)" || { echo "[$SEQ] extract cams $CAMS"; bash "$WS/docker/run_docker.sh" python3 /ws/tools/extract.py "$MDIR" --cams-only "$CAMS"; }
[ "$LIDAR" != multi ] || [ -d "$OFF/livox" ] || { echo "[$SEQ] extract livox"; bash "$WS/docker/run_docker.sh" python3 /ws/tools/extract.py "$MDIR" --livox-only; }

bash "$WS/docker/run_docker.sh" bash -c "
set -e
flock /ws/src/se3-lio/python/.build.lock bash -c \"python3 -c 'import se3_lio.pybind.se3_lio_pybind' 2>/dev/null || { echo '[se3lio] build binding'; pip3 install --no-build-isolation -e /ws/src/se3-lio/python 2>&1 | tail -3; }\"   # 동시 컨테이너가 같은 build/ 를 만지면 import 가 깨진다
python3 /ws/tools/se3lio_run.py '$OFF' '$LIDAR_DIR' /ws/results/$NAME/${SEQ}_imu.tum --config '$CFG' ${IMUDT:+--imu-dt $IMUDT} $SET $LIVOX $CAM $SCAN_DUMP $VIZ 2>&1 | tee /ws/results/$NAME/run.log | grep -E 'se3lio|frames|Error|error' || true
grep -q '^\[se3lio\] done' /ws/results/$NAME/run.log || { echo '[$SEQ] 완주 실패 — run.log 확인'; exit 1; }
python3 /ws/tools/to_prism.py '$OFF/calib.json' /ws/results/$NAME/${SEQ}_imu.tum /ws/results/$NAME/$SEQ.tum
python3 /ws/tools/to_prism.py '$OFF/calib.json' /ws/results/$NAME/${SEQ}_imu.tum /ws/results/$NAME/${SEQ}_g5.tum --grid 0.005 >/dev/null
[ -f '$OFF/gt.tum' ] && cp '$OFF/gt.tum' /ws/results/$NAME/gt.tum && python3 /ws/tools/eval.py /ws/results/$NAME/$SEQ.tum /ws/results/$NAME/gt.tum 2>/dev/null | grep ATE && python3 /ws/tools/eval.py /ws/results/$NAME/${SEQ}_g5.tum /ws/results/$NAME/gt.tum 2>/dev/null | grep ATE || echo '[$SEQ] gt 없음 — 평가 생략'
"
