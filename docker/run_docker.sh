#!/bin/bash
# On the host: bash docker/run_docker.sh [cmd...]   (no cmd: a bash shell)
# The repo is mounted at /ws, the GrandTour data folder ($DATA) at the same path as on the host.
# Resource limits: 8 CPUs and 16 GB (the benchmark environment); CPUSET=<list> pins the container to those cores.
WS="$(realpath "$(dirname "$0")/..")"
MOUNTS=(-v "$WS":/ws)
[ -n "${DATA:-}" ] && [ -d "$DATA" ] && MOUNTS+=(-v "$(realpath "$DATA")":"$(realpath "$DATA")")
RES="$(realpath "$WS/results")"   # results may be a link to another disk: mount the real path at the same path so it resolves inside the container
[ "$RES" != "$WS/results" ] && MOUNTS+=(-v "$RES":"$RES")
TTY=(); [ -t 0 ] && TTY=(-it)

docker run --rm "${TTY[@]}" \
    --cpus="${CPUS:-8}" --memory="${MEM:-16g}" ${CPUSET:+--cpuset-cpus="$CPUSET"} \
    --net=host \
    -e HOME=/tmp \
    ${OMP_NUM_THREADS:+-e OMP_NUM_THREADS="$OMP_NUM_THREADS"} ${SE3LIO_TIMING:+-e SE3LIO_TIMING=1} \
    "${MOUNTS[@]}" \
    -w /ws \
    "${IMAGE:-comfort:ros1}" "${@:-bash}"
