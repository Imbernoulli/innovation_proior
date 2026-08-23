#!/usr/bin/env bash
# Merge a verl FSDP checkpoint actor dir into an HF model, on a dedicated CPU node
# (the login node OOM-kills concurrent merges). Reusable across checkpoints.
#
# Submit:
#   sbatch --export=ALL,CKPT_ACTOR=<.../global_step_N/actor>,TARGET=<.../models/xxx_hf> \
#          slurm/cc_merge_ckpt_cpu.sh

#SBATCH --job-name=cc-merge-ckpt
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$ROOT/verl"
source "$ROOT/.venv-vllm023/bin/activate"
export PYTHONPATH="$ROOT/verl:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

: "${CKPT_ACTOR:?need CKPT_ACTOR}"
: "${TARGET:?need TARGET}"

echo "Merging $CKPT_ACTOR -> $TARGET"
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
