#!/bin/bash
# NatureBench Apptainer pilot — one Slurm job = vLLM (GPU) + eval service + agent (apptainer).
#
# Submit (CPU task, agent mode):
#   sbatch naturebench/slurm/nb_pilot.sh
# Env overrides:
#   TASK=s43588-024-00689-2 MODE=agent|reference MAX_ROUNDS=4 TIMEOUT=2700 ...

#SBATCH --job-name=nb_pilot
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/naturebench/logs/slurm-%j.out

set -euo pipefail

NB=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/naturebench
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
TASK="${TASK:-s43588-024-00689-2}"
MODE="${MODE:-agent}"
BATCH="${BATCH:-pilot_${SLURM_JOB_ID}}"
VLLM_PORT="${VLLM_PORT:-8310}"
EVAL_PORT="${EVAL_PORT:-8321}"
MODEL_PATH="${MODEL_PATH:-$FS/models/Qwen3.5-9B}"
SERVED_NAME="${SERVED_NAME:-qwen35-9b}"
TIMEOUT="${TIMEOUT:-2700}"        # agent budget (gpu-test QoS caps job at 1h)
SETUP_TIMEOUT="${SETUP_TIMEOUT:-1800}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
# task-specific bitrot repair appended after the official Dockerfile RUN cmds
SETUP_EXTRA="${SETUP_EXTRA:-}"

export APPTAINER_CACHEDIR="$NB/containers/.apptainer-cache"
export APPTAINER_TMPDIR="$NB/containers/.apptainer-tmp"

echo "[nb_pilot] task=$TASK mode=$MODE batch=$BATCH node=$(hostname)"

# ---- 1. vLLM server (agent mode only) --------------------------------------
if [ "$MODE" = "agent" ]; then
    echo "[nb_pilot] starting vLLM ($MODEL_PATH as $SERVED_NAME) on :$VLLM_PORT"
    PORT="$VLLM_PORT" SERVED_MODEL_NAME="$SERVED_NAME" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-26624}" MAX_NUM_SEQS=8 \
        nohup bash "$FS/scripts/start_vllm_server.sh" > "$NB/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
    VLLM_PID=$!
    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" | grep -q "$SERVED_NAME"; then
            echo "[nb_pilot] vLLM ready after ${i}0s"
            break
        fi
        sleep 10
        if [ "$i" = "180" ]; then echo "[nb_pilot] FATAL: vLLM never came up"; exit 1; fi
    done
    export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="EMPTY"
fi

# ---- 2. run the harness -----------------------------------------------------
EXTRA=()
if [ "$MODE" = "reference" ]; then
    EXTRA+=(--mode reference --ref-solver "$NB/harness/ref_solver_${TASK}.py")
else
    EXTRA+=(--mode agent --model "$SERVED_NAME")
fi
if [ -n "$SETUP_EXTRA" ]; then
    EXTRA+=(--setup-extra "$SETUP_EXTRA")
fi

python "$NB/harness/nb_run.py" \
    --task "$TASK" \
    --batch "$BATCH" \
    --port "$EVAL_PORT" \
    --timeout "$TIMEOUT" \
    --setup-timeout "$SETUP_TIMEOUT" \
    --max-rounds "$MAX_ROUNDS" \
    --run-timeout "$RUN_TIMEOUT" \
    "${EXTRA[@]}"

RC=$?
echo "[nb_pilot] harness exit=$RC"
[ "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true
exit $RC
