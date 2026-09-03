#!/bin/bash
set -euo pipefail

# Loss family for configs.cifar10_bench. The per-baseline bench_env.sh is
# written by mid_edit / baseline edit_ops INTO the mounted package workspace
# at /workspace/flow-maps/bench_env.sh. common.sh lives under /workspace/_task
# (task dir, read-only) so $_SCRIPT_DIR/bench_env.sh would source a stale task
# copy and every baseline would silently run LSD. Source the workspace copy.
if [[ -f /workspace/flow-maps/bench_env.sh ]]; then
  # shellcheck source=/dev/null
  source /workspace/flow-maps/bench_env.sh
fi
export FLOWMAPS_BENCH_SLURM_ID="${FLOWMAPS_BENCH_SLURM_ID:-0}"

setup_bench_env() {
  cd /workspace/flow-maps
  export PYTHONPATH="/workspace/flow-maps/py:${PYTHONPATH:-}"
  export WANDB_DISABLED=true
  export TF_CPP_MIN_LOG_LEVEL=3
  export SEED="${SEED:-42}"
  export FLOWMAPS_UNET_SIZE="${FLOWMAPS_UNET_SIZE:-medium}"
  export BATCH_SIZE="${BATCH_SIZE:-128}"
  # Paper Boffi et al. (arxiv:2505.18825) trains 400k steps. Official Table 1
  # (CIFAR-10 FID @ NFE 1/2/4/8/16): LSD 8.10/4.37/3.34/3.33/3.57 strictly
  # beats PSD-midpoint and PSD-uniform at every NFE; LSD is the paper's
  # recommended best method, PSD-uniform the weakest. 100k is a 4x-short proxy.
  export MAX_STEPS="${MAX_STEPS:-100000}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
  export NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
  export NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-8}"
  export DS="${DATASET_LOCATION:-/data/tfds}"
}

prepare_cifar10_stats() {
  mkdir -p "$DS" "${OUTPUT_DIR:-./output}" "$DS/cifar10"
  local stats="$DS/cifar10/cifar_stats.npz"
  if [ -f "$stats" ] && [ ! -s "$stats" ]; then
    echo "MLS-Bench: removing empty FID stats (will recompute): $stats"
    rm -f "$stats"
  fi
  if [ ! -f "$stats" ]; then
    echo "Computing CIFAR-10 FID reference stats -> $stats"
    python py/launchers/calc_dataset_fid_stats.py \
      --dataset_location "$DS" \
      --dataset cifar10 \
      --out "$stats" \
      --batch "${FID_STATS_BATCH:-256}"
  fi
}
