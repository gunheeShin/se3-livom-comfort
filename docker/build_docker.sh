#!/bin/bash
# On the host: bash docker/build_docker.sh
cd "$(dirname "$0")" && docker build -t comfort:ros1 .
