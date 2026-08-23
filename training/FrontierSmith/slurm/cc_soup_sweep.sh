#!/usr/bin/env bash
# Soup-alpha sweep to maximize FrontierCS: for each full-FT model x alpha, merge[CPU] --afterok--> FCS eval[GPU].
# Idempotent. Usage: cc_soup_sweep.sh <model_name1,model_name2,...> <alpha1,alpha2,...>
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith; MODELS=$ROOT/models_sft
START=$FS/models/Qwen3.5-9B-bf16; ENV=$ROOT/envs/sft_lf
SOUP=$FS/scripts/cc_model_soup_merge.py; EVAL=$FS/slurm/cc_eval_thinking_both_ailab.sh
MK=$FS/.post_sft; mkdir -p "$MK"
PROCcp="cp -n $START/preprocessor_config.json $START/processor_config.json $START/video_preprocessor_config.json"
IFS=',' read -ra NAMES <<<"$1"; IFS=',' read -ra ALPHAS <<<"$2"
have_full(){ ls "$1"/model*.safetensors 1>/dev/null 2>&1 && [ -e "$1/config.json" ]; }
for name in "${NAMES[@]}"; do
  sft="$MODELS/sft_q35_a100_${name}"; have_full "$sft" || { echo "[wait] $name"; continue; }
  for a in "${ALPHAS[@]}"; do
    pct=$(python3 -c "print(int(round($a*100)))"); soup="$MODELS/soup_q35_a100_${name}_soupa${pct}"; tag="soupr1_${name}_a${pct}"
    { [ -d "$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm" ] || [ -f "$MK/fcs_$tag" ]; } && { echo "[skip] $tag"; continue; }
    if have_full "$soup"; then dep=""; else
      dep=$(sbatch --parsable -J "cc-soup-${name}-a${pct}" --partition=cpu --cpus-per-task=8 --mem=200G --time=01:30:00 \
        -o "$FS/logs/%x-%j.out" -e "$FS/logs/%x-%j.err" \
        --wrap "$ENV/bin/python $SOUP --sft $sft --base $START --alpha $a --out $soup && $PROCcp $soup/" 2>/dev/null)
    fi
    args=(-J "cc-eval-$tag" --export=ALL,MODEL_PATH="$soup",TAG="$tag"); [ -n "$dep" ] && args+=(--dependency=afterok:"$dep")
    sbatch "${args[@]}" "$EVAL" >/dev/null 2>&1 && { touch "$MK/fcs_$tag"; echo "[submit] $tag${dep:+ (afterok:$dep)}"; }
  done
done
