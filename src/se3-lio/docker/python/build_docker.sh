#!/usr/bin/env bash
# Build the python-only image. The image pip-installs the released se3-lio, so it
# needs no repo files at build time — context is this dir (not the repo root).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
docker build -f "$HERE/Dockerfile" -t se3_lio:py "$HERE"
