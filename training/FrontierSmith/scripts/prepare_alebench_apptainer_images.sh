#!/bin/bash
# Pull the minimal ALE-Bench judge images needed by the VERL ALE-lite reward.
#
# Run this on a networked login node before submitting GPU jobs; compute nodes
# do not need internet access when the SIF files are already present.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_DIR="${ALE_BENCH_APPTAINER_DIR:-$PROJECT_ROOT/.cache/apptainer/alebench}"

mkdir -p "$IMAGE_DIR"

apptainer pull --force \
    "$IMAGE_DIR/ale-bench_cpp17-202301.sif" \
    docker://yimjk/ale-bench:cpp17-202301

apptainer pull --force \
    "$IMAGE_DIR/rust_1.79.0-buster.sif" \
    docker://rust:1.79.0-buster

printf 'ALE-Bench Apptainer images are ready in %s\n' "$IMAGE_DIR"
