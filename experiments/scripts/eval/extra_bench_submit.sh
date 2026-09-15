#!/usr/bin/env bash
# extra_bench_submit.sh -- re-run the three non-main evaluation families (idea generation,
# taste, research judgment) for the base and ft01mix lines, WITH a year-bearing system prompt.
#
# Why re-run at all: gen_client.py had GEN_SYSTEM_PROMPT but never had it set by any submitter,
# and idea_client.py / judge_pairwise.py / judge_pointwise.py had no system-prompt mechanism at
# all. So every taste and research-judgment number on disk was produced with a bare user prompt
# and no year, while the main benches all ran with "It is now year 2026. You are a good
# researcher." That is the protocol mismatch, not a nuance.
#
# New runs land under TAG=<arm>_y26sp so the old no-system-prompt results stay intact as the
# comparison baseline; nothing is overwritten.
#
#   bash extra_bench_submit.sh [gen|idea|ideav2|judge3|all] [arm ...]
set -uo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
SYSP="It is now year 2026. You are a good researcher."   # byte-identical to the main-bench prompt
LOG=$D/logs/extra_bench_jobids.txt

declare -A MP=(
  [base9b_v2c]="$FS/models/Qwen3.5-9B-bf16"
  [ft01mix_a10]="$D/models/full_wd01_withag_v2b_soup10"
  [rlv5_base_s20]="$D/models/rlv5_base_s20"
  [rlv5_ft01mix_a10_s20]="$D/models/rlv5_ft01mix_a10_s20"
  [base4b]="$D/models/Qwen3.5-4B"
  [4b_ft01mix_a10]="$D/models/4b_full_wd01_withag_soup10"
  [rlv5_4b_base_s20]="$D/models/rlv5_4b_base_s20"
  [rlv5_4b_ft01mix_a10_s20]="$D/models/rlv5_4b_ft01mix_a10_s20"
)
WHICH="${1:-all}"; shift || true
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=("${!MP[@]}")

go() {  # family wrapper walltime jobprefix extra-env
  local fam=$1 wrap=$2 wall=$3 pre=$4 extra=$5 arm M TAG j
  for arm in "${ARMS[@]}"; do
    M="${MP[$arm]:?unknown arm $arm}"
    [ -f "$M/config.json" ] || { echo "no model $M for $arm"; continue; }
    # TAGSUF lets an off-protocol run be kept as a labelled control instead of being overwritten.
    # y26sp   = year-2026 system prompt, but sampled WITHOUT presence_penalty (pre-alignment).
    # y26pp   = same prompt, sampled on the RL rollout protocol (presence_penalty=1.5).
    TAG="${arm}_${TAGSUF:-y26sp}"
    local ex="${extra//@TAG@/$TAG}"        # OUT_DIR needs the per-arm tag; sbatch --export does no expansion
    [ -d "$D/logs/locks/${pre}-${TAG}.lock" ] && { echo "SKIP ${pre}-${TAG} (lock)"; continue; }
    j=$(sbatch --parsable --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G \
        --time="$wall" --job-name="${pre}-${TAG}" \
        --output="$D/logs/%x-%j.out" --error="$D/logs/%x-%j.out" \
        --export=ALL,"MODEL=$M,TAG=$TAG,GEN_SYSTEM_PROMPT=$SYSP,IDEA_SYSTEM_PROMPT=$SYSP,JUDGE_SYSTEM_PROMPT=$SYSP,$ex" \
        "$D/slurm_overlay/$wrap") || { echo "sbatch failed $pre $arm"; continue; }
    echo "$(date +%F_%T) $fam $arm job=$j tag=$TAG" | tee -a "$LOG"
  done
}

case "$WHICH" in
  gen)    go gen    cc_eval_gen_ailab.sh  10:00:00 gen    "OUT_DIR=$D/outputs/cc_gen_@TAG@" ;;
  idea)   go idea   cc_eval_idea_ailab.sh 08:00:00 idea32 "TASKS_FILE=$D/ideabench/tasks.jsonl,MAX_TOKENS=32768,OUT_DIR=$D/outputs/cc_idea32k_@TAG@" ;;
  ideav2) go ideav2 cc_eval_idea_ailab.sh 08:00:00 ideav2 "TASKS_FILE=$D/ideabench/tasks_v2.jsonl,MAX_TOKENS=32768,OUT_DIR=$D/outputs/cc_ideav2_@TAG@" ;;
  judge3) go judge3 cc_eval_idea_ailab.sh 08:00:00 judge3 "TASKS_FILE=$D/ideabench/tasks_v3.jsonl,MAX_TOKENS=32768,OUT_DIR=$D/outputs/cc_judge3_@TAG@" ;;
  all) for w in gen idea ideav2 judge3; do bash "$0" "$w" "${ARMS[@]}"; done ;;
  *) echo "usage: extra_bench_submit.sh [gen|idea|ideav2|judge3|all] [arm ...]"; exit 2;;
esac
