#!/bin/bash
# On the host: HRUN_FROM=comfort-docker hrun --mem 16 --cpus 8 --gpu 0 bash docker/build_docker.sh
cd "$(dirname "$0")" && docker build -t comfort:ros1 .
