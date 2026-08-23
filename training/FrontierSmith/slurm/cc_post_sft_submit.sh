#!/usr/bin/env bash
# One-shot, idempotent post-SFT submitter (replaces the fragile polling loop).
# For each r1 cell whose SFT model is ready and whose eval isn't done/submitted:
#   full-FT -> soup-average(alpha) [CPU] --afterok--> FCS eval [GPU] ; also eval RAW
#   LoRA    -> merge(s=1.0) [CPU]        --afterok--> FCS eval [GPU]
# Re-run anytime (e.g. after methodtraj finishes); skips anything already done/submitted via markers.
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith; MODELS=$ROOT/models_sft
START=$FS/models/Qwen3.5-9B-bf16; ENV=$ROOT/envs/sft_lf
SOUP=$FS/scripts/cc_model_soup_merge.py; LORA=$ROOT/merge_lora_scaled.py
EVAL=$FS/slurm/cc_eval_thinking_both_ailab.sh
ALPHA="${SOUP_ALPHA:-0.5}"; APCT=$(python3 -c "print(int(round($ALPHA*100)))")
MK=$FS/.post_sft; mkdir -p "$MK"
PROCcp="cp -n $START/preprocessor_config.json $START/processor_config.json $START/video_preprocessor_config.json"

have_full(){ ls "$1"/model*.safetensors 1>/dev/null 2>&1 && [ -e "$1/config.json" ]; }
have_adapter(){ [ -e "$1/adapter_config.json" ] && [ -e "$1/adapter_model.safetensors" ]; }
eval_done(){ [ -d "$FS/outputs/cc_eval_${1}_thinking_32k_both_vllm" ] || [ -f "$MK/fcs_$1" ]; }
submit_eval(){ # dir tag [depjid]
  local d="$1" t="$2" dep="${3:-}"
  eval_done "$t" && { echo "  [skip eval] $t (done/submitted)"; return; }
  local args=(-J "cc-eval-$t" --export=ALL,MODEL_PATH="$d",TAG="$t")
  [ -n "$dep" ] && args+=(--dependency=afterok:"$dep")
  sbatch "${args[@]}" "$EVAL" >/dev/null 2>&1 && { touch "$MK/fcs_$t"; echo "  [eval] $t${dep:+ (afterok:$dep)}"; }
}
submit_merge(){ # cpu merge; echoes jobid (or empty if out already exists / submit fails)
  local cmd="$1" out="$2" name="$3"
  have_full "$out" && { echo ""; return; }
  sbatch --parsable -J "cc-merge-$name" --partition=cpu --cpus-per-task=8 --mem=200G --time=01:30:00 \
    -o "$FS/logs/%x-%j.out" -e "$FS/logs/%x-%j.err" --wrap "$cmd" 2>/dev/null
}

CELLS=( "method_r1|full|-" "methodv4_r1|full|-" "methodtraj_r1|full|-" "methodtraj_v4_r1|full|-"
        "method_r1|lora|32" "methodv4_r1|lora|32" "methodtraj_r1|lora|32" "methodtraj_v4_r1|lora|32"
        "methodv4_r1|lora|64" "methodtraj_v4_r1|lora|64" )

for cell in "${CELLS[@]}"; do
  IFS='|' read -r name kind rank <<<"$cell"
  if [ "$kind" = full ]; then
    sft="$MODELS/sft_q35_a100_${name}"; have_full "$sft" || { echo "[wait] full $name (not trained)"; continue; }
    submit_eval "$sft" "sftr1_${name}"                                   # raw
    soup="$MODELS/soup_q35_a100_${name}_soupa${APCT}"
    if have_full "$soup"; then submit_eval "$soup" "soupr1_${name}_a${APCT}"
    else
      jid=$(submit_merge "$ENV/bin/python $SOUP --sft $sft --base $START --alpha $ALPHA --out $soup && $PROCcp $soup/" "$soup" "soup_${name}")
      [ -n "$jid" ] && { echo "  [merge] soup_${name} jid=$jid"; submit_eval "$soup" "soupr1_${name}_a${APCT}" "$jid"; }
    fi
  else
    ad="$MODELS/lora_q35_a100_${name}_r${rank}"; have_adapter "$ad" || { echo "[wait] lora ${name}_r${rank} (not trained)"; continue; }
    merged="$MODELS/merged_lora_q35_a100_${name}_r${rank}_s10"
    if have_full "$merged"; then submit_eval "$merged" "lorar1_${name}_r${rank}"
    else
      jid=$(submit_merge "$ENV/bin/python $LORA --base $START --adapter $ad --scale 1.0 --out $merged --proc-src $START" "$merged" "lora_${name}_r${rank}")
      [ -n "$jid" ] && { echo "  [merge] lora_${name}_r${rank} jid=$jid"; submit_eval "$merged" "lorar1_${name}_r${rank}" "$jid"; }
    fi
  fi
done
echo "done."
