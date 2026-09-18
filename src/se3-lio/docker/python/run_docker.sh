#!/usr/bin/env bash
# Run the python-only image with X11 + GPU so the Polyscope viewer can display.
# Edit DATA_DIR to the host directory holding your bags.
DATA_DIR=/path/to/bags          # edit: host dir with ROS1 .bag / ROS2 rosbag dirs

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
xhost +local:root >/dev/null 2>&1

docker run -it --rm \
    --gpus all \
    -e DISPLAY="$DISPLAY" \
    -e XDG_RUNTIME_DIR=/tmp/xdg \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$REPO_ROOT":/ws/src/SE3-LIO \
    -v "$DATA_DIR":/ws/data \
    -w /ws/src/SE3-LIO \
    se3_lio:py bash

# Inside the container:
#   mkdir -p /tmp/xdg
#   se3_lio_pipeline /ws/data/eee_01/eee_01.bag \
#       --params ros/ros1/config/ntu.yaml --input-type ros1-ouster \
#       --max-frames 1500 --visualize
