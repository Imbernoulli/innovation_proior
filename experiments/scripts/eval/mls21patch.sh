#!/usr/bin/env bash
# mls21patch.sh -- re-run ONLY the two tasks that used to be impossible.
#
# Why this exists: for every arm, `causal-observational-linear-gaussian` and
# `optimization-multi-objective` came back `status=agent_failed, score=None,
# settings=[]`. That was NOT the model. The driver crashed in the agent
# constructor, before a single token was sent:
#
#   agent/base.py:43  load_mid_edit_ops -> tools.py:5124 exec_module
#     tasks/<task>/edits/mid_edit.py  ->  import dgp   # host-only
#       holdout/causal-observational-linear-gaussian/dgp.py:19
#         from causallearn.graph.Dag import Dag  -> ModuleNotFoundError
#       holdout/optimization-multi-objective/dgp.py:38
#         from deap import benchmarks           -> ModuleNotFoundError
#
# The container side was never broken: vendor/images/{causal-learn,deap}.sif and
# vendor/pkg_configs/{causal-learn,deap} both exist. Only the HOST driver python
# (/home/zy7019/miniconda3) lacked the libs, because holdout/*/dgp.py is host-only
# scoring code that is deliberately not shipped into the container.
#
# Fixed 2026-09-02 by installing what the pkg_configs declare, at the container's
# versions -- causal-learn 0.1.4.4 from vendor/external_packages/causal-learn
# (copied to $D/vendor_src because setup.py builds in-place and st3812's tree is
# read-only), and deap==1.4.1. Nothing already installed was upgraded.
#
# The driver has no per-task resume, so re-running a full 21-task sweep to recover
# two tasks would cost ~2 h/arm. Instead this writes a 2-task run to a separate
# OUTPUT_BASE and mls_merge.py stitches it onto the existing cc_mls21_<TAG>.
#
#   bash scripts/mls21patch.sh <TAG> [<TAG> ...]
set -euo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith

TASKS2="causal-observational-linear-gaussian optimization-multi-objective"

CONC="${CONCURRENCY:-2}"
TMO="${TASK_TIMEOUT:-7200}"

# Deterministic ports collide when two of our jobs share a node -- that is how
# mls21-rlv5_ft03nm_a20_s15 (13358221) died with EADDRINUSE. Spread them out.
port_for() { echo $(( 41000 + (RANDOM % 20000) )); }

cd "$FS"   # the script takes PROJECT_ROOT from SLURM_SUBMIT_DIR; --chdir does not override it
for TAG in "$@"; do
  [ -d "$D/models/$TAG" ] || { echo "SKIP $TAG: no merged model" >&2; continue; }
  jid=$(sbatch --parsable --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 \
    --mem=200G --time=04:00:00 --job-name="mls2p-$TAG" \
    --output="$D/logs/%x-%j.out" --error="$D/logs/%x-%j.out" \
    --export=ALL,MODEL_PATH="$D/models/$TAG",TAG="$TAG",OUTPUT_BASE="$D/outputs/cc_mls2p_$TAG",\
MLSBENCH_ROOT="$D/mlsroot",HF_HOME="$D/.hf",VLLM_VENV="$D/envs/vllm023",\
VLLM_CACHE_DIR="$D/.cache/vllm",EVAL_RESEARCHER_YEAR=2026,\
MLSBENCH_SYS_PREFIX="It is now year 2026.",CONCURRENCY="$CONC",TASK_TIMEOUT="$TMO",VLLM_PORT="$(port_for)",\
TASKS="$TASKS2" \
    "$FS/slurm/cc_eval_mlsbench_cpu_ailab.sh")
  echo "mls2p-$TAG -> $jid  (2 tasks, out=cc_mls2p_$TAG)"
done
