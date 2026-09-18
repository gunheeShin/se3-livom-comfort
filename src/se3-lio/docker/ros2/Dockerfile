FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git locales software-properties-common curl \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get upgrade -y \
    && apt-get install -y ros-humble-desktop ros-humble-pcl-ros python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git \
    && cd Livox-SDK2 && mkdir build && cd build && cmake .. && make -j$(($(nproc)-1)) && make install \
    && cd / && rm -rf Livox-SDK2

# Pin to last pre-Jazzy master (2024-09-11); newer master (13eb05e) drops the
# HUMBLE_ROS code path and breaks the Humble colcon build (LIVOX_INTERFACES_* NOTFOUND).
RUN mkdir -p /ws/src \
    && cd /ws/src && git clone https://github.com/Livox-SDK/livox_ros_driver2.git \
    && cd livox_ros_driver2 && git checkout 6b9356c \
    && cp package_ROS2.xml package.xml

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd /ws && colcon build --symlink-install --cmake-args -DROS_EDITION=ROS2 -DHUMBLE_ROS=humble"

# --- Python (rclpy) node: install the released binding ---
# ros2/scripts/se3_lio_node.py imports the compiled `se3_lio` package (cp310
# manylinux wheel; the C++ core + OpenMP are bundled). Bump the pin + rebuild to
# update. For local dev against unreleased code, override at runtime:
#   pip3 install --no-build-isolation -e ./python/
RUN pip3 install --no-cache-dir "se3-lio>=0.2"

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "source /ws/install/setup.bash" >> ~/.bashrc \
    && echo 'alias cb="colcon build --symlink-install"' >> ~/.bashrc \
    && echo 'alias si="source install/setup.bash"' >> ~/.bashrc \
    && echo 'export PS1="\[\e[32m\]$(cat /proc/self/cgroup | grep docker | tr "/" "\n" | tail -1 | cut -c1-12)\[\e[34m\]:\[\e[36m\]\w\[\e[0m\]\$ "' >> ~/.bashrc
