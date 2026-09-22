FROM ros:noetic-perception

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=noetic

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake ninja-build \
    python3-pip python3-dev \
    libeigen3-dev libboost-all-dev libomp-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# GTSAM 4.1 (pose graph of the online backend)
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:borglab/gtsam-release-4.1 \
    && apt-get update && apt-get install -y --no-install-recommends libgtsam-dev libgtsam-unstable-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Sophus non-templated (a621ff2), the version FAST-LIVO2 requires. so2.cpp fails with an lvalue error on recent Eigen, so it is patched.
# This version's make install does not install headers or SophusConfig.cmake, so source and build dirs stay under /opt and only the config is registered.
RUN git clone https://github.com/strasdat/Sophus.git /opt/Sophus \
    && cd /opt/Sophus && git checkout a621ff2e56c56c839a6c40418d42c3c254424b5c \
    && sed -i 's/unit_complex_.real() = 1.;/unit_complex_.real(1.);/; s/unit_complex_.imag() = 0.;/unit_complex_.imag(0.);/' sophus/so2.cpp \
    && mkdir build && cd build && cmake .. && make -j$(($(nproc)-1)) && make install \
    && mkdir -p /usr/local/lib/cmake/Sophus && cp SophusConfig.cmake /usr/local/lib/cmake/Sophus/ \
    && ldconfig

# se3-lio Python binding source build + rosbag handling
RUN pip3 install --no-cache-dir "pip>=21.3,<25" \
    && pip3 install --no-cache-dir scikit-build-core pybind11 ninja "cmake<4" \
       numpy "pydantic>=2" pyyaml tqdm "rosbags>=0.9" evo

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

WORKDIR /ws
