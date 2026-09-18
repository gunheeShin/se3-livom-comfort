DATA_DIR=/path/to/rosbag2_data    # edit: host dir containing your ROS2 bag(s)

docker run -it --rm \
    --gpus all \
    --shm-size 64G \
    --cpus=$(nproc) \
    --net=host \
    --ipc=host \
    --pid=host \
    -e ROS_DOMAIN_ID=10 \
    -e DISPLAY=unix$DISPLAY \
    --env=DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus" \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(realpath ../..):/ws/src/SE3-LIO \
    -v $DATA_DIR:/ws/data/ \
    -w /ws \
    se3_lio:ros2
