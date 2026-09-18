#!/bin/bash
# 호스트에서: bash docker/run_docker.sh [cmd...]   (cmd 없으면 bash 셸)
# IMAGE=comfort:hba 처럼 환경변수로 다른 이미지를 고른다(기본 comfort:ros1).
# comfort_ws → /ws, GrandTour SSD는 호스트와 같은 경로로 마운트. 빌드는 컨테이너 안에서 catkin build.
WS="$(realpath "$(dirname "$0")/..")"
DATA=/media/gunhee/gun_T7_17/Research/LIO/PublichDataset/grandtour
MOUNTS=(-v "$WS":/ws)
[ -d "$DATA" ] && MOUNTS+=(-v "$DATA":"$DATA")
RES="$(realpath "$WS/results")"   # results 는 외장 SSD 로 가는 링크 — 컨테이너에서도 풀리게 실경로를 같은 경로로 마운트
[ "$RES" != "$WS/results" ] && MOUNTS+=(-v "$RES":"$RES")
TTY=(); [ -t 0 ] && TTY=(-it)

docker run --rm "${TTY[@]}" \
    --net=host \
    -e DISPLAY="$DISPLAY" \
    -e QT_X11_NO_MITSHM=1 \
    -e HOME=/tmp \
    ${OMP_NUM_THREADS:+-e OMP_NUM_THREADS="$OMP_NUM_THREADS"} ${SE3LIO_TIMING:+-e SE3LIO_TIMING=1} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    "${MOUNTS[@]}" \
    -w /ws \
    "${IMAGE:-comfort:ros1}" "${@:-bash}"
