#!/usr/bin/env bash
# Merge a verl FSDP checkpoint actor dir into an HF model, on a CPU node.
#
# Overlay of fsroot/slurm/cc_merge_ckpt_cpu.sh. That script hardcodes
#   ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
#   source "$ROOT/.venv-vllm023/bin/activate"
# and bl3615's venv has no `tensordict`, so `python -m verl.model_merger` dies with
# ModuleNotFoundError before doing any work. Same failure class as the MLSBENCH_PY /
# MLSBENCH_ROOT / VLLM_VENV defaults: an inherited path tuned for another account.
# Here everything points at zy7019-owned copies instead.
#
# Submit:
#   sbatch --export=ALL,CKPT_ACTOR=<.../global_step_N/actor>,TARGET=<.../models/xxx_hf> \
#          slurm_overlay/cc_merge_ckpt_cpu_zy7.sh

#SBATCH --job-name=cc-merge-ckpt-zy7
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=02:00:00

set -euo pipefail

D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
VERL_ROOT="${VERL_ROOT:-$D/fsroot/verl}"
VENV="${VENV:-$D/envs/rl}"

cd "$VERL_ROOT"
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# These checkpoints bake attn_implementation=flash_attention_2 into config.json, and
# transformers refuses to dispatch FA2 on CPU ("FlashAttention2 is not available on
# CPU"). So the merge needs a visible GPU even though the work itself is state-dict
# shuffling. Set ALLOW_GPU=1 and submit to a GPU partition.
if [ "${ALLOW_GPU:-0}" != "1" ]; then
  export CUDA_VISIBLE_DEVICES=""
fi

: "${CKPT_ACTOR:?need CKPT_ACTOR}"
: "${TARGET:?need TARGET}"

echo "[merge] python=$(command -v python)"
echo "[merge] $CKPT_ACTOR -> $TARGET"

python -m verl.model_merger merge --backend fsdp \
  --local_dir "$CKPT_ACTOR" \
  --target_dir "$TARGET" \
  --trust-remote-code

echo "=== result ==="
ls -la "$TARGET"/model*.safetensors 2>/dev/null | awk '{print $5,$NF}'
python - "$TARGET" <<'PY'
import json,sys,os,glob
d=sys.argv[1]
idx=os.path.join(d,"model.safetensors.index.json")
if os.path.exists(idx):
    ks=json.load(open(idx))["weight_map"]
    print("keys",len(ks),"has A_log",any('A_log' in k for k in ks),"config",os.path.exists(os.path.join(d,'config.json')))
else:
    print("single-shard?", glob.glob(os.path.join(d,'*.safetensors')))
PY
