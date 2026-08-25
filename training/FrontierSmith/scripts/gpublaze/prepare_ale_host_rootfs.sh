#!/usr/bin/env bash
# One-time: export the ALE-Bench judge-image rootfs for the NO-DOCKER host
# backend (scripts/gpublaze/pysite/ale_host_backend.py). Docker is used ONCE
# here (create+export+rm); the scoring path itself never touches dockerd.
#
#   bash scripts/gpublaze/prepare_ale_host_rootfs.sh [tag ...]
#
# Default tag: cpp17-202301 (the only language image the reward/eval chain
# uses). Output: $ALE_BENCH_CACHE/host-rootfs/<tag>/ (default
# ~/.cache/ale-bench/host-rootfs/<tag>/). Requires ~3GB free per tag.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_gpublaze.sh"

TAGS=("${@:-cpp17-202301}")
if [ "${#TAGS[@]}" -eq 1 ] && [ -z "${1:-}" ]; then TAGS=(cpp17-202301); fi
ROOT="${ALE_BENCH_HOST_ROOTFS:-${ALE_BENCH_CACHE:-$FS_ROOT/.cache/ale-bench}/host-rootfs}"
mkdir -p "$ROOT"

for tag in "${TAGS[@]}"; do
  image="ale-bench:${tag}"
  dest="$ROOT/${tag}"
  tmp="$dest.tmp.$$"
  echo "[rootfs] exporting $image -> $dest"
  docker image inspect "$image" >/dev/null 2>&1 || { echo "FATAL: image $image not pulled" >&2; exit 1; }
  cname="ale-host-rootfs-export-$$"
  rm -rf "$tmp"; mkdir -p "$tmp"
  cid="$(docker create --name "$cname" "$image")"
  trap 'docker rm -f "$cname" >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT
  docker export "$cid" -o "$tmp/rootfs.tar"
  docker rm "$cname" >/dev/null
  tar -xf "$tmp/rootfs.tar" -C "$tmp"
  rm -f "$tmp/rootfs.tar"
  [ -f "$tmp/usr/bin/bash" ] || { echo "FATAL: exported rootfs has no /usr/bin/bash" >&2; exit 1; }
  rm -rf "$dest"
  mv "$tmp" "$dest"
  trap - EXIT
  echo "[rootfs] OK: $dest ($(du -sh "$dest" | cut -f1))"
done
