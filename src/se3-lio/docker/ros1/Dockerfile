FROM nvidia/cuda:11.8.0-base-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=noetic

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release build-essential git cmake \
    python3-pip python3-dev \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop ros-noetic-pcl-ros \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool python3-catkin-tools \
    && rosdep init && rosdep update \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git \
    && cd Livox-SDK2 && mkdir build && cd build && cmake .. && make -j$(($(nproc)-1)) && make install

RUN mkdir -p /ws/src \
    && cd /ws/src && git clone https://github.com/Livox-SDK/livox_ros_driver2.git \
    && cp /ws/src/livox_ros_driver2/package_ROS1.xml /ws/src/livox_ros_driver2/package.xml

RUN pip uninstall netifaces -y && pip install netifaces

# --- Python (rospy) node: install the released binding ---
# ros1/scripts/se3_lio_node.py imports the compiled `se3_lio` package. Install the
# released manylinux wheel (cp38 bundles the C++ core + OpenMP) — no cmake/build
# toolchain needed on this old (20.04, cmake 3.16) image. Bump the pin + rebuild
# to pull a newer release. For local dev against unreleased code, build from the
# mounted source at runtime instead:
#   pip3 install "pip>=21.3,<25" scikit-build-core pybind11 ninja "cmake<4"
#   pip3 install --no-build-isolation -e ./python/     # editable = in-tree ../cpp core
RUN pip3 install --no-cache-dir "se3-lio>=0.2"

RUN source /opt/ros/noetic/setup.bash \
    && cd /ws \
    && catkin build --cmake-args -DROS_EDITION=ROS1

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc \
    && echo "source /ws/devel/setup.bash" >> ~/.bashrc \
    && echo 'alias cb="catkin build"' >> ~/.bashrc \
    && echo 'alias sd="source devel/setup.bash"' >> ~/.bashrc \
    && echo 'export PS1="\[\e[32m\]$(cat /proc/self/cgroup | grep docker | tr "/" "\n" | tail -1 | cut -c1-12)\[\e[34m\]:\[\e[36m\]\w\[\e[0m\]\$ "' >> ~/.bashrc
