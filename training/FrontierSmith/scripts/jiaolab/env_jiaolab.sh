#!/usr/bin/env bash
# jiaolab portability layer -- common environment for the FrontierSmith eval
# stack on the single-node 8xA100-80G-PCIe machine `jiaolab` (no slurm, docker
# present but requires sudo => UNUSABLE, apptainer 1.3.1 available, NO bwrap).
#
# DESIGN RULE (same as scripts/gpublaze/env_gpublaze.sh): the historical slurm/
# and scripts/ files keep their Princeton-final semantics; everything
# machine-specific lives HERE and is injected via the env vars those scripts
# already honour. No historical script -- and no gpublaze script -- is edited.
#
# Machine deltas vs gpublaze, in one place:
#   docker (sudo-only)      -> apptainer SIF backend for ALE-Bench judging
#   no bwrap                -> the gpublaze `host` ALE backend is NOT usable here
#   GPUs shared with `druv` -> fs_guard_gpus refuses any card with < FS_MIN_FREE_GB
#                              free (gpublaze's static GPU 6/7 ban has no analogue)
#   A100 sm80 (no NVLink)   -> one TP=1 engine per card; never TP=2
#
# Source this from every jiaolab wrapper:  source "$(dirname "$0")/env_jiaolab.sh"

FS_JIAOLAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FS_ROOT="${FS_ROOT:-$(cd "$FS_JIAOLAB_DIR/../.." && pwd)}"
export FS_NODE_NAME="${FS_NODE_NAME:-jiaolab}"

# ---- discovery registry (was: shared GPFS dir; now: local disk) ---------------
export VLLM_POOL_REGISTRY="${VLLM_POOL_REGISTRY:-$FS_ROOT/.cache/vllm_pool}"

# ---- model caches --------------------------------------------------------------
export HF_HOME="${HF_HOME:-/home/bohan/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ---- python environments -------------------------------------------------------
# Serve side: vLLM 0.21.0 + torch 2.11 cu128 (sm80-compatible), WITH the
# FrontierSmith penalty fast path applied (scripts/jiaolab/
# apply_vllm_penalty_fastpath_jiaolab.sh). Without the fast path, presence_penalty
# 1.5 costs ~6x throughput -- verify the "[fs] vLLM penalty fast path ACTIVE"
# banner in every serve log.
export VLLM_VENV="${VLLM_VENV:-/home/bohan/venv-vllm-jiaolab}"
# Client side: CPU-only venv (torch-cpu + verl editable + ALE-Bench editable).
export FS_CLIENT_VENV="${FS_CLIENT_VENV:-$FS_ROOT/.venv-jiaolab}"

# ---- judge ---------------------------------------------------------------------
# UNLIKE gpublaze, the REAL go-judge binary CANNOT run here: Ubuntu 24.04 /
# kernel 6.8 enforces AppArmor's apparmor_restrict_unprivileged_userns, so
# `unshare -U true` passes but writing /proc/self/uid_map is denied and go-judge
# dies with "fork/exec /proc/self/exe: permission denied" (also inside apptainer).
# Fixing that requires sudo => out of scope. jiaolab therefore judges FrontierCS
# with the certified gojudge_shim_v2 (Princeton-authenticated 94.3% byte-exact).
# The binary is kept on disk so the day sudo/AppArmor changes, GJ_BACKEND=gojudge
# just works.
export GO_JUDGE_BIN="${GO_JUDGE_BIN:-$FS_ROOT/.cache/bin/go-judge}"
export GJ_BACKEND="${GJ_BACKEND:-auto}"   # auto probes `unshare -Ur` => shim here
# Node judge app stage dir (node_modules live here; node v18 on this box).
export FS_JUDGE_STAGE_DIR="${FS_JUDGE_STAGE_DIR:-$FS_ROOT/.cache/jiaolab/judge_app}"

# ---- ALE-Bench: apptainer backend ----------------------------------------------
# docker needs sudo here and bwrap does not exist, so the ONLY sandbox is
# apptainer. SIFs live in $ALE_BENCH_APPTAINER_DIR with the names ALE-Bench's
# own _apptainer_image_path() expects (ale-bench_<tag>.sif / rust_<tag>.sif).
export ALE_BENCH_APPTAINER_DIR="${ALE_BENCH_APPTAINER_DIR:-/home/bohan/sif}"
export ALE_BENCH_CACHE="${ALE_BENCH_CACHE:-$FS_ROOT/.cache/ale-bench}"
export ALE_BENCH_DATA="${ALE_BENCH_DATA:-$FS_ROOT/data/alebench/local_data}"

# ---- GPU police ----------------------------------------------------------------
# jiaolab's 8 A100s are SHARED with user `druv` (2-4G per card, ~20G on GPU 0).
# NEVER kill another user's process; only schedule onto cards with enough FREE
# memory for a whole TP=1 engine. Guard refuses anything below FS_MIN_FREE_GB.
export FS_MIN_FREE_GB="${FS_MIN_FREE_GB:-70}"
# Optional hard ban list (comma separated), empty by default on this box.
export FS_FORBID_GPUS="${FS_FORBID_GPUS:-}"

_fs_free_gb() {  # _fs_free_gb <gpu_index> -> free GiB on stdout
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null \
    | awk -F', *' '{printf "%d", ($1-$2)/1024}'
}

fs_guard_gpus() {
  local g free rc=0
  IFS=',' read -ra _gs <<< "${1:?fs_guard_gpus needs a GPU list}"
  for g in "${_gs[@]}"; do
    case ",$FS_FORBID_GPUS," in
      *",$g,"*) echo "REFUSING GPU $g: in FS_FORBID_GPUS" >&2; rc=1; continue ;;
    esac
    free="$(_fs_free_gb "$g")"
    if [ -z "$free" ]; then
      echo "REFUSING GPU $g: nvidia-smi gave no reading" >&2; rc=1; continue
    fi
    if [ "$free" -lt "$FS_MIN_FREE_GB" ]; then
      echo "REFUSING GPU $g: only ${free}G free (< FS_MIN_FREE_GB=${FS_MIN_FREE_GB}G)." >&2
      echo "  jiaolab GPUs are shared with user druv -- never evict another user's process." >&2
      echo "  Pick a freer card, or lower FS_MIN_FREE_GB deliberately." >&2
      rc=1; continue
    fi
    echo "[gpu-guard] GPU $g ok (${free}G free)" >&2
  done
  return $rc
}

# fs_pick_free_gpus <n> -> comma-separated list of the n freest usable GPUs.
fs_pick_free_gpus() {
  local want="${1:-1}" i free out=()
  for i in $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); do
    case ",$FS_FORBID_GPUS," in *",$i,"*) continue ;; esac
    free="$(_fs_free_gb "$i")"
    [ -n "$free" ] && [ "$free" -ge "$FS_MIN_FREE_GB" ] && out+=("$free:$i")
  done
  [ "${#out[@]}" -ge "$want" ] || { echo "only ${#out[@]} GPU(s) with >=${FS_MIN_FREE_GB}G free, need $want" >&2; return 1; }
  printf '%s\n' "${out[@]}" | sort -t: -k1,1nr | head -n "$want" | cut -d: -f2 | paste -sd,
}

mkdir -p "$VLLM_POOL_REGISTRY" "$FS_ROOT/logs" "$FS_ROOT/outputs"
