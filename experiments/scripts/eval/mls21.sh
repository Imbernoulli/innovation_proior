#!/usr/bin/env bash
# mls21.sh -- submit an MLS-Bench run that every arm can be compared against.
#
# User directive 2026-09-02: drop llm-scaling-law-discovery (it agent_failed in 100% of
# runs and contributes nothing), leaving 21 tasks, and GUARANTEE all 21 finish so every
# arm is averaged over the SAME denominator.
#
# What was wrong before: CONCURRENCY=20 ran all 22 tasks in parallel against a single
# vLLM on a single GPU. Per-task median elapsed ranged 2537 s -> 4703 s across runs and
# timeouts tracked it (corr 0.66); timeouts anticorrelated with the reported score
# (corr -0.49). The tasks that timed out were precisely the ones carrying the score
# (ml-anomaly-detection 0.502, optimization-evolution-strategy 0.487,
# ml-clustering-algorithm 0.388, ml-selective-deferral 0.376, ml-active-learning 0.366),
# so a starved run did not score lower -- it lost the scoring tasks outright, and
# mean_score silently averaged over 16 tasks for one arm and 20 for another.
# Worst case seen: base_s10 and ft03nm_s10 were submitted 7 s apart, landed on the SAME
# node, and ft03nm_s10 came back with 7 timeouts and the worst score of any run.
#
# Fix: CONCURRENCY=7 (3 waves of 21) so each task gets ~3x the backend throughput, a
# 2 h per-task ceiling, and a 12 h job so 3 full waves fit even in the worst case.
# Submit these one per node and do not co-schedule two of them.
#
#   bash scripts/mls21.sh <TAG> [<TAG> ...]
set -euo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

TASKS21="causal-discovery-discrete causal-observational-linear-gaussian \
causal-observational-linear-non-gaussian causal-observational-nonlinear \
causal-treatment-effect ml-active-learning ml-anomaly-detection ml-calibration \
ml-clustering-algorithm ml-dimensionality-reduction ml-ensemble-boosting \
ml-missing-data-imputation ml-selective-deferral ml-subgroup-calibration-shift \
ml-symbolic-regression mlsys-moe-load-balance optimization-evolution-strategy \
optimization-hyperparameter-search optimization-multi-objective optimization-nas \
optimization-online-bandit"

CONC="${CONCURRENCY:-7}"
TMO="${TASK_TIMEOUT:-7200}"

# cc_eval_mlsbench_cpu_ailab.sh defaults to VLLM_PORT=34000+JOBU, which collides whenever
# two of our jobs share a node and their ids land on the same offset -- that is how
# mls21-rlv5_ft03nm_a20_s15 (13358221) died after 1h08m with
#   OSError: [Errno 98] Address already in use  /  vLLM exited early
# while ev-rlv5_base_s15-r0 was on the same node. Spread the ports out instead.
port_for() { echo $(( 41000 + (RANDOM % 20000) )); }

cd "$FS"   # the script takes PROJECT_ROOT from SLURM_SUBMIT_DIR; --chdir does not override it
for TAG in "$@"; do
  [ -d "$D/models/$TAG" ] || { echo "SKIP $TAG: no merged model" >&2; continue; }
  jid=$(sbatch --parsable --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 \
    --mem=200G --time=12:00:00 --job-name="mls21-$TAG" \
    --output="$D/logs/%x-%j.out" --error="$D/logs/%x-%j.out" \
    --export=ALL,MODEL_PATH="$D/models/$TAG",TAG="$TAG",OUTPUT_BASE="$D/outputs/cc_mls21_$TAG",\
MLSBENCH_ROOT="$D/mlsroot",HF_HOME="$D/.hf",VLLM_VENV="$D/envs/vllm023",\
VLLM_CACHE_DIR="$D/.cache/vllm",EVAL_RESEARCHER_YEAR=2026,\
MLSBENCH_SYS_PREFIX="It is now year 2026.",CONCURRENCY="$CONC",TASK_TIMEOUT="$TMO",VLLM_PORT="$(port_for)",\
TASKS="$TASKS21" \
    "$FS/slurm/cc_eval_mlsbench_cpu_ailab.sh")
  echo "mls21-$TAG -> $jid  (21 tasks, concurrency=$CONC, per-task ${TMO}s, out=cc_mls21_$TAG)"
done
