#!/usr/bin/env bash
# gpublaze portability layer -- common environment for the FrontierSmith eval
# stack on the single-node 8xH100 machine `gpublaze` (no slurm, no apptainer,
# rootless docker 29.x with NVIDIA CDI).
#
# DESIGN RULE (docs/FRONTIERSMITH_TRAINING_EVAL_INTEGRATION_zh.md "复现提醒"):
# the historical slurm/ and scripts/ files keep their Princeton-final semantics;
# everything machine-specific lives HERE and is injected via the env vars those
# scripts already honour. No historical script is edited.
#
# Source this from every gpublaze wrapper:   source "$(dirname "$0")/env_gpublaze.sh"

# Resolve the FrontierSmith root from this file's location (scripts/gpublaze/..)
FS_GPUBLAZE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FS_ROOT="${FS_ROOT:-$(cd "$FS_GPUBLAZE_DIR/../.." && pwd)}"

# ---- discovery registry (was: shared GPFS dir; now: local disk) ---------------
export VLLM_POOL_REGISTRY="${VLLM_POOL_REGISTRY:-$FS_ROOT/.cache/vllm_pool}"

# ---- model caches --------------------------------------------------------------
# The machine's real HF cache (has Qwen3.6-27B / Qwen3.8-27B); NOT the
# project-local .cache/huggingface the Princeton scripts default to.
export HF_HOME="${HF_HOME:-/home/bohanlyu/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---- python environments -------------------------------------------------------
# Serve side: reuse the machine's existing vLLM env (vllm 0.21.0, torch 2.11 cu128)
# instead of building a second multi-GB GPU env on the 99%-full /srv.
export VLLM_VENV="${VLLM_VENV:-/srv/home/bohanlyu/sesl/.venv}"
# Client side: CPU-only venv built by this port (torch-cpu + verl editable +
# ALE-Bench editable). $FS_ROOT/.venv is a symlink to it so historical scripts
# that hardcode .venv keep working.
export FS_CLIENT_VENV="${FS_CLIENT_VENV:-$FS_ROOT/.venv-gpublaze}"

# ---- judge ---------------------------------------------------------------------
# userns works on gpublaze (`unshare -U true` passes), so the REAL go-judge
# binary is usable; the certified python shim remains the fallback.
export GO_JUDGE_BIN="${GO_JUDGE_BIN:-$FS_ROOT/.cache/bin/go-judge}"
export GJ_BACKEND="${GJ_BACKEND:-auto}"

# ---- GPU police ----------------------------------------------------------------
# GPUs 6,7 belong to user zzh's long-running sglang -- NEVER schedule on them.
# Wrappers take GPUS=<list>; this guard refuses 6/7 unless FS_ALLOW_GPU67=1.
fs_guard_gpus() {
  local g
  IFS=',' read -ra _gs <<< "${1:?fs_guard_gpus needs a GPU list}"
  for g in "${_gs[@]}"; do
    if { [ "$g" = "6" ] || [ "$g" = "7" ]; } && [ "${FS_ALLOW_GPU67:-0}" != "1" ]; then
      echo "REFUSING GPU $g: reserved for another user's service (set FS_ALLOW_GPU67=1 only if that has changed)" >&2
      return 1
    fi
  done
}

mkdir -p "$VLLM_POOL_REGISTRY" "$FS_ROOT/logs" "$FS_ROOT/outputs"
