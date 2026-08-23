#!/usr/bin/env bash
# Post-SFT pipeline for the remediated (r1) matrix: per finished SFT model ->
#   full-FT: eval RAW + soup-average(alpha) ; LoRA: merge(s=1.0) + eval.
# Each derived model gets FCS-Algorithm(thinking, GPU) + MLS-devfix(CPU) evals.
# Idempotent (marker files in $FS/.post_sft) + backfill loop. Re-runnable.
#   DO_MLS=1 SOUP_ALPHA=0.5 bash cc_post_sft_pipeline.sh
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS=$ROOT/FrontierSmith
MODELS=$ROOT/models_sft
START=$FS/models/Qwen3.5-9B-bf16
ENV=$ROOT/envs/sft_lf
SOUP_PY=$FS/scripts/cc_model_soup_merge.py
LORA_PY=$ROOT/merge_lora_scaled.py
MLS_DEV=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev
SOUP_ALPHA="${SOUP_ALPHA:-0.5}"; APCT=$(python3 -c "print(int(round($SOUP_ALPHA*100)))")
DO_MLS="${DO_MLS:-1}"
RETRY_S="${RETRY_S:-180}"; MAX_LOOPS="${MAX_LOOPS:-120}"
MARK=$FS/.post_sft; mkdir -p "$MARK"
LOG=$FS/logs/post_sft_pipeline_$(date +%Y%m%d_%H%M%S).log
PROC=(preprocessor_config.json processor_config.json video_preprocessor_config.json)

# name|kind|rank   (rank '-' for full)
CELLS=( "method_r1|full|-" "methodv4_r1|full|-" "methodtraj_r1|full|-" "methodtraj_v4_r1|full|-"
        "method_r1|lora|32" "methodv4_r1|lora|32" "methodtraj_r1|lora|32" "methodtraj_v4_r1|lora|32"
        "methodv4_r1|lora|64" "methodtraj_v4_r1|lora|64" )

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
have_weights(){ [ -e "$1/config.json" ] && ls "$1"/model*.safetensors "$1"/*.safetensors 1>/dev/null 2>&1; }
have_adapter(){ [ -e "$1/adapter_config.json" ] && [ -e "$1/adapter_model.safetensors" ]; }
sft_busy(){ squeue -u "$USER" -h -n "$1" -t RUNNING,PENDING -o "%i" 2>/dev/null | grep -q .; }
copy_proc(){ for f in "${PROC[@]}"; do [ -e "$START/$f" ] && cp -n "$START/$f" "$1/" 2>/dev/null; done; }

submit_fcs(){ # dir tag
  local d="$1" t="$2" mk="$MARK/fcs_$2"
  [ -f "$mk" ] && return
  [ -d "$FS/outputs/cc_eval_${t}_thinking_32k_both_vllm" ] && { touch "$mk"; return; }
  have_weights "$d" || return
  copy_proc "$d"
  sbatch --job-name="cc-eval-$t" --export=ALL,MODEL_PATH="$d",TAG="$t" \
    "$FS/slurm/cc_eval_thinking_both_ailab.sh" >>"$LOG" 2>&1 && { touch "$mk"; log "FCS submitted $t"; }
}
submit_mls(){ # dir tag
  [ "$DO_MLS" = 1 ] || return
  local d="$1" t="${2}_devfix" mk="$MARK/mls_$2"
  [ -f "$mk" ] && return
  [ -d "$FS/outputs/cc_mlsbench_cpu_${t}" ] && { touch "$mk"; return; }
  have_weights "$d" || return
  copy_proc "$d"
  sbatch --parsable --job-name="cc-mlsdevfix-$t" --time=08:00:00 \
    --export=ALL,MODEL_PATH="$d",TAG="$t",MLSBENCH_ROOT="$MLS_DEV",EVAL_RESEARCHER_YEAR=2025 \
    "$FS/slurm/cc_eval_mlsbench_cpu_ailab.sh" >>"$LOG" 2>&1 && { touch "$mk"; log "MLS submitted $t"; }
}
submit_merge(){ # type sftdir out [scale]   -> CPU job; returns 0 only when OUT is ready
  local type="$1" sft="$2" out="$3" scale="${4:-}" mk="$MARK/merge_$(basename "$out")"
  have_weights "$out" && return 0
  [ -f "$mk" ] && return 1
  local cmd
  if [ "$type" = soup ]; then
    have_weights "$sft" || return 1
    copy_proc "$sft"
    cmd="$ENV/bin/python $SOUP_PY --sft $sft --base $START --alpha $SOUP_ALPHA --out $out"
  else
    have_adapter "$sft" || return 1
    cmd="$ENV/bin/python $LORA_PY --base $START --adapter $sft --scale $scale --out $out --proc-src $START"
  fi
  sbatch --job-name="cc-merge-$(basename "$out")" --partition=cpu --cpus-per-task=8 --mem=200G --time=01:30:00 \
    --output="$FS/logs/%x-%j.out" --error="$FS/logs/%x-%j.err" --wrap="$cmd" >>"$LOG" 2>&1 \
    && { touch "$mk"; log "merge submitted $(basename "$out")"; }
  return 1
}

log "post-SFT pipeline start (SOUP_ALPHA=$SOUP_ALPHA -> a$APCT, DO_MLS=$DO_MLS)"
for i in $(seq 1 "$MAX_LOOPS"); do
  alldone=1
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r name kind rank <<<"$cell"
    if [ "$kind" = full ]; then
      sft="$MODELS/sft_q35_a100_${name}"
      sft_busy "sft_q35_${name}" && { alldone=0; continue; }
      have_weights "$sft" || { alldone=0; continue; }
      # raw
      submit_fcs "$sft" "sftr1_${name}"; submit_mls "$sft" "sftr1_${name}"
      # soup-average
      soup="$MODELS/soup_q35_a100_${name}_soupa${APCT}"
      if submit_merge soup "$sft" "$soup"; then
        submit_fcs "$soup" "soupr1_${name}_a${APCT}"; submit_mls "$soup" "soupr1_${name}_a${APCT}"
      else alldone=0; fi
    else
      ad="$MODELS/lora_q35_a100_${name}_r${rank}"
      sft_busy "lora_q35_${name}_r${rank}" && { alldone=0; continue; }
      have_adapter "$ad" || { alldone=0; continue; }   # adapter dir has adapter_config.json + adapter_model.safetensors
      merged="$MODELS/merged_lora_q35_a100_${name}_r${rank}_s10"
      if submit_merge lora "$ad" "$merged" 1.0; then
        submit_fcs "$merged" "lorar1_${name}_r${rank}"; submit_mls "$merged" "lorar1_${name}_r${rank}"
      else alldone=0; fi
    fi
  done
  [ "$alldone" = 1 ] && { log "ALL post-SFT eval submitted"; break; }
  sleep "$RETRY_S"
done
log "post-SFT pipeline loop end"
