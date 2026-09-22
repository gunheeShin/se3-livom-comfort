#!/bin/bash
# One-mission SE(3)-LIO run: check extract -> (build the binding if missing) -> se3lio_run -> prism conversion -> evaluation.
# On the host: bash scripts/run_se3lio.sh <seq> [--tag NAME] [--lidar multi] [--config PATH] [--set KEY=VAL ...]
#   --set KEY=VAL: override an SE3LIOConfig field (max_iter=10). May be repeated
#   --imu-dt S:     IMU time offset (s); default = the <seq> entry of src/se3-lio/config/imu_dt.yaml
#   Output: results/<seq>-se3lio[-<tag>]/{<seq>_imu.tum,<seq>.tum,gt.tum,run.log,provenance/}
#   --lidar multi:  read comfort_offline/lidar/ + livox/ separately and let the se3-lio core merge them (no deduplication). Without livox/: extract --livox-only
#   The camera is on by default: image stamps are the epoch boundaries + photometric update (visual_en=1). Result name gets -livo (--livo is kept for old calls, same as default)
#   --online-bag:   read the mission folder's bags directly at 1x without extraction (the public default path). Bit-identical trajectory to the offline run with the same options.
#                   Result name gets -online; results/<name>/calib.json (built from the bag) and latency.csv (pose delay vs image time) are written. Hesai (+ --lidar multi) + one camera
#   --lio-only:     camera off (LiDAR scan times are the epochs). Result name has no -livo
#   --cam:          use image stamps as epoch boundaries only, no photometric update. Result name gets -cam
#   --cams A,B:     camera list (default front_center). Several: name gets -<N>cam. Without cam_<name>/: extract --cams-only
#   --img-dt S:     image time offset (s, se3lio_run --img-time-offset)
#   --rt:           measured profile: OMP 8 threads, result name gets -rt. The default profile is OMP 4 threads. Either way timing.csv is written
#   --stage-timing: log the three core stages (predict, update, map ms) as TIMING lines in run.log (SE3LIO_TIMING=1). update = LiDAR + camera update, map = voxel + visual map, leaf/inl = downsample grid and LiDAR inlier count of that scan
#   --backend:      online backend: every 4 keyframes (25 scans each, 10 s) a global plane BA over everything so far from the LIO poses, fed into an iSAM2 pose graph (no window BA).
#                   Nothing runs at the end; the estimate at that moment goes to results/<name>/<seq>_backend.tum (IMU) and _backend_prism.tum, rounds to backend_rounds.csv, name gets -backend
#   --backend-set K=V,...: backend parameters (voxel_size=1.0,downsample_size=0.2,eigen_ratio=0.01,reject_ratio=0.05,max_iter=10,layers=3,threads=8,every=4,hess_const=a:b:c:d:e:f,gravity_sigma_deg=0.02). The gravity factor puts the accelerometer mean of a stationary chunk (4 s) on the stationary node, relative to the LIO bias and gravity (0 = off)
#   --omp N:        LIO OMP thread count instead of the profile value (rt 8, internal 4). Used to share cores with the backend worker (threads)
set -euo pipefail
WS="$(realpath "$(dirname "$0")/..")"
DATA="${DATA:?set DATA to the folder that holds the GrandTour mission folders}"
SEQ="$1"; shift; ARGS="$*"
TAG=""; IMUDT=""; LIDAR=""; CFG=/ws/src/se3-lio/config/comfort.yaml; SET=""; CAM=1; LIVO=1; IMGDT=""; CAMS=front_center; RT=""; OB=""; BACKEND=""; BACKENDSET=""; OMP=""
while [ $# -gt 0 ]; do case "$1" in --tag) TAG="-$2"; shift;; --imu-dt) IMUDT="$2"; shift;; --lidar) LIDAR="$2"; shift;; --config) CFG="$2"; shift;; --set) SET="$SET --set $2"; shift;; --cam) CAM=1; LIVO="";; --lio-only) CAM=""; LIVO="";; --online-bag) OB=1;; --cams) CAMS="$2"; shift;; --img-dt) IMGDT="$2"; shift;; --rt) RT=1;; --stage-timing) export SE3LIO_TIMING=1;; --livo) CAM=1; LIVO=1;; --backend) BACKEND=1;; --backend-set) BACKENDSET="$2"; shift;; --omp) OMP="$2"; shift;; esac; shift; done

[ -n "$IMUDT" ] || IMUDT=$(awk -v s="$SEQ:" '$1 == s {print $2}' "$WS/src/se3-lio/config/imu_dt.yaml")
[ -n "$IMUDT" ] || { echo "[$SEQ] no --imu-dt and no entry in src/se3-lio/config/imu_dt.yaml"; exit 2; }
MDIR=$(ls -d "$DATA"/*_"${SEQ^^}"_release_* | head -1)
OFF="$MDIR/comfort_offline"
LIDAR_DIR="$OFF/lidar"; LIVOX=""
[ "$LIDAR" = multi ] && LIVOX="--livox-dir $OFF/livox"
MODE=""; [ -z "$CAM" ] || MODE="-cam"; [ -z "$LIVO" ] || MODE="-livo"
NCAM=$(tr , "\n" <<<"$CAMS" | wc -l); [ "$NCAM" -le 1 ] || MODE="$MODE-${NCAM}cam"
NAME="$SEQ-se3lio${LIDAR:+-$LIDAR}$MODE$TAG${BACKEND:+-backend}${RT:+-rt}${OB:+-online}"
export OMP_NUM_THREADS=$([ -n "$RT" ] && echo 8 || echo 4)
[ -z "$OMP" ] || export OMP_NUM_THREADS=$OMP   # --omp N: LIO thread count different from the profile (sharing 8 cores with the backend worker)
SET="--set visual_en=${LIVO:-0}$SET"   # the yaml default is on, so it is always set explicitly. A later user --set wins
CAMARG="--cams none"; [ -z "$CAM" ] || CAMARG="--cams $CAMS${IMGDT:+ --img-time-offset $IMGDT}"
OUT="$WS/results/$NAME"
mkdir -p "$OUT"

PROV="$OUT/provenance"; mkdir -p "$PROV"
git -C "$WS" rev-parse HEAD > "$PROV/commit"
{ git -C "$WS" diff --binary HEAD
  git -C "$WS" ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do git -C "$WS" diff --binary --no-index /dev/null "$f" || true; done
} > "$PROV/diff.patch"
{ echo "date: $(date -Is)"; echo "cmd: $0 $SEQ $ARGS"; echo "docker: cpus=${CPUS:-8} mem=${MEM:-16g} cpuset=${CPUSET:-}"
  echo "image: $(docker image inspect --format '{{.Id}}' "${IMAGE:-comfort:ros1}")"; echo "lidar_dir: $LIDAR_DIR"; echo "config: $CFG"; echo "set: $SET"; echo "backend: ${BACKEND:+1 $BACKENDSET}"; echo "OMP_NUM_THREADS=$OMP_NUM_THREADS rt=${RT:-0}"; } > "$PROV/run.txt"

SRC="'$OFF' '$LIDAR_DIR'"; CALIB="$OFF/calib.json"
if [ -n "$OB" ]; then SRC="'$MDIR' - --online-bag"; CALIB="/ws/results/$NAME/calib.json"; else   # online-bag needs no extraction
[ -f "$OFF/manifest.json" ] || { echo "[$SEQ] extract"; bash "$WS/docker/run_docker.sh" python3 /ws/tools/extract.py "$MDIR" --cam $([ -n "$CAM" ] && echo front_center || echo none); }
[ -z "$CAM" ] || [ -d "$OFF/cam" ] || { echo "[$SEQ] $OFF/cam missing: run extract with --cam front_center"; exit 1; }
[ -z "$CAM" ] || python3 -c "import json,sys; c=json.load(open('$OFF/calib.json')); sys.exit(0 if all(n in c.get('cams',{}) for n in '$CAMS'.split(',')) else 1)" || { echo "[$SEQ] extract cams $CAMS"; bash "$WS/docker/run_docker.sh" python3 /ws/tools/extract.py "$MDIR" --cams-only "$CAMS"; }
[ "$LIDAR" != multi ] || [ -d "$OFF/livox" ] || { echo "[$SEQ] extract livox"; bash "$WS/docker/run_docker.sh" python3 /ws/tools/extract.py "$MDIR" --livox-only; }
fi

bash "$WS/docker/run_docker.sh" bash -c "
set -e
flock /ws/src/se3-lio/python/.build.lock bash -c \"python3 -c 'import se3_lio.pybind.se3_lio_pybind' 2>/dev/null || { echo '[se3lio] build binding'; pip3 install --no-build-isolation -e /ws/src/se3-lio/python 2>&1 | tail -3; }\"   # concurrent containers touching the same build/ break the import
python3 /ws/tools/se3lio_run.py $SRC /ws/results/$NAME/${SEQ}_imu.tum --config '$CFG' ${IMUDT:+--imu-dt $IMUDT} $SET $LIVOX $CAMARG ${BACKEND:+--backend $BACKENDSET} 2>&1 | tee /ws/results/$NAME/run.log | grep -E 'se3lio|frames|Error|error' || true
grep -q '^\[se3lio\] done' /ws/results/$NAME/run.log || { echo '[$SEQ] run did not finish, see run.log'; exit 1; }
python3 /ws/tools/to_prism.py '$CALIB' /ws/results/$NAME/${SEQ}_imu.tum /ws/results/$NAME/$SEQ.tum
[ -f '$OFF/gt.tum' ] && cp '$OFF/gt.tum' /ws/results/$NAME/gt.tum && python3 /ws/tools/eval.py /ws/results/$NAME/$SEQ.tum /ws/results/$NAME/gt.tum 2>/dev/null | grep ATE || echo '[$SEQ] no gt, evaluation skipped'
[ -z '$BACKEND' ] || { python3 /ws/tools/to_prism.py '$CALIB' /ws/results/$NAME/${SEQ}_backend.tum /ws/results/$NAME/${SEQ}_backend_prism.tum; [ -f '$OFF/gt.tum' ] && python3 /ws/tools/eval.py /ws/results/$NAME/${SEQ}_backend_prism.tum /ws/results/$NAME/gt.tum 2>/dev/null | grep ATE | sed 's/^/[backend] /'; true; }
"
