#!/bin/bash
# On the host: bash docker/run_docker.sh [cmd...]   (no cmd: a bash shell)
# IMAGE=comfort:dev selects another image (default comfort:ros1).
# The repo is mounted at /ws, the GrandTour data folder at the same path as on the host.
WS="$(realpath "$(dirname "$0")/..")"
DATA=/media/gunhee/gun_T7_17/Research/LIO/PublichDataset/grandtour
MOUNTS=(-v "$WS":/ws)
[ -d "$DATA" ] && MOUNTS+=(-v "$DATA":"$DATA")
RES="$(realpath "$WS/results")"   # results is a link to an external SSD; mount the real path at the same path so it resolves inside the container
[ "$RES" != "$WS/results" ] && MOUNTS+=(-v "$RES":"$RES")
TTY=(); [ -t 0 ] && TTY=(-it)

docker run --rm "${TTY[@]}" \
    --net=host \
    -e DISPLAY="$DISPLAY" \
    -e QT_X11_NO_MITSHM=1 \
    -e HOME=/tmp \
    ${OMP_NUM_THREADS:+-e OMP_NUM_THREADS="$OMP_NUM_THREADS"} ${SE3LIO_TIMING:+-e SE3LIO_TIMING=1} ${SE3LIO_DUMP:+-e SE3LIO_DUMP=1} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    "${MOUNTS[@]}" \
    -w /ws \
    "${IMAGE:-comfort:ros1}" "${@:-bash}"
