# Python-only image — run the se3_lio Python package + CLI (offline pipeline,
# benchmark, Polyscope/Rerun viz). No ROS.
#
# Installs the RELEASED wheel by default (C++ core + OpenMP bundled). The build
# toolchain is kept, so to test unreleased LOCAL changes, override at runtime with
# an editable source build:
#   docker run -v <repo>:/ws/src/SE3-LIO ... se3_lio:py
#   pip install --no-build-isolation -e ./python/
#
# Build (context = this dir, no repo files needed):  bash docker/python/build_docker.sh
# For a CPU-only / headless image, swap the base for ubuntu:22.04 (software GL).
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
ENV NVIDIA_DRIVER_CAPABILITIES=all

# GL/X11 runtime for Polyscope + a compile toolchain kept so a runtime editable
# install (pip install -e ./python) can still build the core for local dev.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      build-essential cmake git ninja-build libgomp1 \
      libgl1 libglu1-mesa libxext6 libxrender1 \
      libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir scikit-build-core pybind11 ninja

# Released wheel (C++ core bundled) + Polyscope/Rerun viewers. Bump the pin + rebuild to update.
RUN pip install --no-cache-dir "se3-lio[viz,rerun]>=0.2"

WORKDIR /ws
CMD ["bash"]
