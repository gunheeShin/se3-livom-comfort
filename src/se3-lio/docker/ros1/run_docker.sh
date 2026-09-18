docker run -it --rm \
    --gpus all \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(realpath ../..):/ws/src/SE3-LIO \
    -w /ws \
    se3_lio:ros1
