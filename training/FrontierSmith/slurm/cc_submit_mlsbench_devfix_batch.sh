#!/usr/bin/env bash
# Submit the dev-fix MLS-Bench CPU re-run for the method-only + methodtraj-only + start
# model set, against the FIXED MLS-Bench-dev root (fix/oracle-leakage-pilot).
#
# Each job: full 20 CPU tasks, max_tests=3 (eval-script default), researcher year 2026.
# TAG carries a _devfix suffix so outputs/cc_mlsbench_cpu_<TAG>/ never clobbers old results.
# MLSBENCH_ROOT is passed EXPLICITLY in --export (not relying on ALL) so the job binds the
# fixed code + the dev-root vendor symlink farm regardless of resubmit context.
set -euo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$ROOT"

MLSBENCH_ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev
EVAL_RESEARCHER_YEAR=2026
EVAL_SCRIPT=slurm/cc_eval_mlsbench_cpu_ailab.sh

# model_dir  TAG
JOBS=(
  "$ROOT/models/Qwen3.5-9B-bf16|q35_start_devfix"
  "/scratch/gpfs/CHIJ/bohan/fs/models_sft/sft_q35_a100_method|q35_a100_method_sft_devfix"
  "/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_a100_method_soupa10|q35_a100_method_soup10_devfix"
  "/scratch/gpfs/CHIJ/bohan/fs/models_sft/sft_q35_a100_methodtraj|q35_a100_methodtraj_sft_devfix"
  "/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_a100_methodtraj_soupa10|q35_a100_methodtraj_soup10_devfix"
)

for spec in "${JOBS[@]}"; do
  MD="${spec%%|*}"; TAG="${spec##*|}"
  if [ ! -e "$MD/config.json" ]; then echo "SKIP (no config.json): $MD" >&2; continue; fi
  JID=$(sbatch --parsable \
    --job-name="cc-mlsdevfix-${TAG}" \
    --time="08:00:00" \
    --export="ALL,MODEL_PATH=${MD},TAG=${TAG},MLSBENCH_ROOT=${MLSBENCH_ROOT},EVAL_RESEARCHER_YEAR=${EVAL_RESEARCHER_YEAR}" \
    "$EVAL_SCRIPT")
  echo "submitted job=$JID TAG=$TAG MODEL=$MD"
  echo "  summary: outputs/cc_mlsbench_cpu_${TAG}/summary.json"
done
