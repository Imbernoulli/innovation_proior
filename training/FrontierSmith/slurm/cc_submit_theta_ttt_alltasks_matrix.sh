#!/usr/bin/env bash
# Submit the FULL {on-scope models} x {all CPU tasks} matrix for BOTH benchmarks:
#   - ThetaEvolve  : circle_packing_modular, first/second/third_autocorr_inequality, hadamard_matrix  (5 CPU tasks)
#   - TTT-Discover : first/second/third_autocorr_inequality, circle_packing_modular                   (4 CPU tasks,
#                    run through the SAME OpenEvolve offline adapter; TTT native Tinker loop unavailable)
#
# Each job = 1 GPU (vLLM serves the model; the evolutionary VERIFIER is pure CPU/numpy).
# GPU verifier tasks (TTT gpu_mode) are SKIPPED. erdos_min_overlap / denoising / ahc have no offline
# OpenEvolve adapter and are BLOCKED (see report).
#
# Jobs queue with normal priority and will schedule as GPUs free behind the protected jobs,
# respecting the ailab 16-GPU/user cap (we do NOT cancel anything).
#
# Usage:
#   bash slurm/cc_submit_theta_ttt_alltasks_matrix.sh           # submit all
#   DRYRUN=1 bash slurm/cc_submit_theta_ttt_alltasks_matrix.sh  # print only
set -euo pipefail
FS_ROOT=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
THETA_LAUNCH="$FS_ROOT/slurm/cc_eval_theta_openevolve_ailab.sh"
TTT_LAUNCH="$FS_ROOT/slurm/cc_eval_ttt_discover_openevolve_ailab.sh"
ITERATIONS="${ITERATIONS:-12}"
SEED="${SEED:-3407}"
DRYRUN="${DRYRUN:-0}"

# model tag -> absolute dir
declare -A MODELS=(
  [q35_start]="$FS_ROOT/models/Qwen3.5-9B"
  [q35_method_sft]="/scratch/gpfs/CHIJ/bohan/fs/models_sft/sft_q35_a100_method"
  [q35_method_soup10]="/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_a100_method_soupa10"
  [q35_methodtraj_sft]="/scratch/gpfs/CHIJ/bohan/fs/models_sft/sft_q35_a100_methodtraj"
  [q35_methodtraj_soup10]="/scratch/gpfs/CHIJ/bohan/fs/models_sft/soup_q35_a100_methodtraj_soupa10"
)
# preserve a stable model order
MODEL_ORDER=(q35_start q35_method_sft q35_method_soup10 q35_methodtraj_sft q35_methodtraj_soup10)

THETA_TASKS=(circle_packing_modular first_autocorr_inequality second_autocorr_inequality third_autocorr_inequality hadamard_matrix)
TTT_TASKS=(first_autocorr_inequality second_autocorr_inequality circle_packing_modular)  # third_autocorr REMOVED: not a TTT-Discover task (no AC3)

# distinct port per job avoids vLLM port collisions across co-scheduled jobs
# IMPORTANT: vLLM (when VLLM_PORT is set) scans UP TO 5 ports UPWARD from VLLM_PORT
# for its internal distributed/zmq coordination (vllm/utils/network_utils.py
# get_open_ports_list count=5). If two of our jobs co-locate on one node and their
# serving ports are <=5 apart, one job's internal-port scan grabs the other's
# serving port -> torch.distributed EADDRINUSE on init (observed: jobs on
# della-i19g1 with ports 8300/8301 died binding 8303). FIX: space serving ports by
# a wide STRIDE (>= scan window + buffer) AND jitter the base per invocation so
# resubmits don't systematically re-collide.
PORT_STRIDE="${PORT_STRIDE:-20}"
# random base in [8200, 9800) on a STRIDE grid, unless caller pins PORT_BASE
if [ -z "${PORT_BASE:-}" ]; then
  PORT_BASE=$(( 8200 + (RANDOM % 80) * PORT_STRIDE ))
fi
port=$PORT_BASE
JOBLIST="$FS_ROOT/logs/theta_ttt_matrix_jobs_$(date +%Y%m%d_%H%M%S).tsv"
mkdir -p "$FS_ROOT/logs"
echo -e "jobid\tbench\tmodel_tag\ttask\tmodel_dir\tport" > "$JOBLIST"

# A combo is "done/queued" if a summary.json exists OR a matching job is already
# in the queue. Used by SKIP_EXISTING=1 so the script is idempotent and safe to
# re-run on a loop to backfill jobs rejected by the QOS submit cap.
already_present() { # bench mtag task
  local bench="$1" mtag="$2" task="$3" tag outdir jname
  if [ "$bench" = "theta" ]; then tag="$mtag"; jname="cc-eval-theta"; else tag="ttt_$mtag"; jname="cc-eval-ttt-math-$mtag"; fi
  outdir="/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/outputs/cc_eval_theta_${tag}_${task}"
  # finished?
  if compgen -G "$outdir/job_*/summary.json" >/dev/null 2>&1; then return 0; fi
  # already queued/running? (match this exact tag+task in our squeue job names)
  if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$jname"; then
    # job-name alone isn't task-specific for theta; fall back to a marker file
    if [ -f "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/.submitted_${bench}_${mtag}_${task}" ]; then return 0; fi
  fi
  [ -f "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/.submitted_${bench}_${mtag}_${task}" ] && return 0
  return 1
}

submit() { # bench launcher model_tag task model_dir
  local bench="$1" launcher="$2" mtag="$3" task="$4" mdir="$5"
  local jid out
  if [ "${SKIP_EXISTING:-0}" = "1" ] && already_present "$bench" "$mtag" "$task"; then
    echo "skip (already present): $bench $mtag $task"; return
  fi
  # NOTE: theta launcher HAS #SBATCH directives -> invoke with sbatch.
  #       ttt launcher is a plain wrapper that does `exec sbatch ...` and
  #       re-tags to ttt_<tag> itself -> invoke with bash. Both get <mtag>.
  if [ "$DRYRUN" = "1" ]; then
    if [ "$bench" = "theta" ]; then
      echo "[DRY] theta $mtag $task port=$port -> sbatch $launcher $mdir $mtag $task $ITERATIONS"
    else
      echo "[DRY] ttt   $mtag $task port=$port -> bash $launcher $mdir $mtag $task $ITERATIONS (-> tag ttt_$mtag)"
    fi
    return
  fi
  if [ "$bench" = "theta" ]; then
    jid=$(SEED="$SEED" VLLM_PORT="$port" sbatch --parsable \
          "$launcher" "$mdir" "$mtag" "$task" "$ITERATIONS" 2>&1) \
      || { echo "SUBMIT FAILED ($bench $mtag $task): $jid" >&2; return; }
    jid=$(echo "$jid" | grep -oE '[0-9]+' | tail -1)
  else
    out=$(SEED="$SEED" VLLM_PORT="$port" bash "$launcher" "$mdir" "$mtag" "$task" "$ITERATIONS" 2>&1) \
      || { echo "SUBMIT FAILED ($bench $mtag $task): $out" >&2; return; }
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1)
  fi
  if [ -z "$jid" ]; then echo "SUBMIT REJECTED ($bench $mtag $task) — likely QOS submit cap; will retry" >&2; return 1; fi
  touch "/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/.submitted_${bench}_${mtag}_${task}"
  echo -e "${jid}\t${bench}\t${mtag}\t${task}\t${mdir}\t${port}" >> "$JOBLIST"
  echo "submitted $bench $mtag $task -> job $jid (port $port)"
}

for mtag in "${MODEL_ORDER[@]}"; do
  mdir="${MODELS[$mtag]}"
  for t in "${THETA_TASKS[@]}"; do submit theta "$THETA_LAUNCH" "$mtag" "$t" "$mdir"; port=$((port+PORT_STRIDE)); done
  for t in "${TTT_TASKS[@]}";  do submit ttt   "$TTT_LAUNCH"   "$mtag" "$t" "$mdir"; port=$((port+PORT_STRIDE)); done
done

echo "=== job list -> $JOBLIST ==="
[ "$DRYRUN" = "1" ] || cat "$JOBLIST"
