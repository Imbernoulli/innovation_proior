#!/usr/bin/env bash
# MLS-Bench-dev (oracle-leak-free) batch for the remediated r1 models. Each = 1 GPU, 8h.
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith; cd "$ROOT"
MLSBENCH_ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev
JOBS=(
  "$ROOT/models/Qwen3.5-9B-bf16|r1_start_devfix"
  "/scratch/gpfs/CHIJ/bohan/fs/models_sft/sft_q35_a100_methodv4_r1|r1_sft_methodv4_devfix"
  "/scratch/gpfs/CHIJ/bohan/fs/models_sft/sft_q35_a100_methodtraj_v4_r1|r1_sft_methodtraj_v4_devfix"
  "/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_a100_methodv4_r1_soupa50|r1_soup_methodv4_a50_devfix"
  "/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_a100_methodtraj_v4_r1_soupa50|r1_soup_methodtraj_v4_a50_devfix"
)
for spec in "${JOBS[@]}"; do
  MD="${spec%%|*}"; TAG="${spec##*|}"
  [ -e "$MD/config.json" ] || { echo "SKIP (no model): $MD"; continue; }
  [ -d "outputs/cc_mlsbench_cpu_${TAG}" ] && { echo "SKIP (done): $TAG"; continue; }
  JID=$(sbatch --parsable --job-name="cc-mlsdevfix-${TAG}" --time=08:00:00 \
    --export="ALL,MODEL_PATH=${MD},TAG=${TAG},MLSBENCH_ROOT=${MLSBENCH_ROOT},EVAL_RESEARCHER_YEAR=2026" \
    slurm/cc_eval_mlsbench_cpu_ailab.sh)
  echo "MLS submitted job=$JID TAG=$TAG"
done
