#!/usr/bin/env bash
# Self-healing watchdog for the cc_pipeline.sh chains. Every CHECK_S (default 2h):
#  (1) submit the deferred matched RL-after eval once a variant's HF export lands,
#  (2) AUTO-RECOVER any FAILED FCS/MLS eval (re-submit on the port-strided eval script),
#  (3) write a clean STATUS table,
#  (4) EXIT+notify ONLY on a real train/soup/RL stage failure, or when all done.
# cc-rlexport "FAILED" is NOT a problem if the HF landed (its internal 26624 eval can collide; we run a
# separate matched eval). Eval failures are auto-recovered, not alerted.
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs; FS=$ROOT/FrontierSmith; START=$FS/models/Qwen3.5-9B-bf16
VARIANTS="${VARIANTS:-methodv4_r2 methodtraj_v4_r2}"; APCT="${APCT:-30}"
CHECK_S="${CHECK_S:-7200}"; MAX_LOOPS="${MAX_LOOPS:-60}"
PROC="cp -n $START/preprocessor_config.json $START/processor_config.json $START/video_preprocessor_config.json"
STATUS=$FS/logs/pipeline_status.txt
have_full(){ ls "$1"/model*.safetensors 1>/dev/null 2>&1 && [ -e "$1/config.json" ]; }
jstate(){ sacct -X -n --name="$1" -S now-3days -o State 2>/dev/null | tail -1 | tr -d ' '; }
isfail(){ [[ "$1" == FAILED* || "$1" == TIMEOUT* || "$1" == OUT_OF_ME* || "$1" == NODE_FAIL* || "$1" == CANCELLED* ]]; }
fcs(){ local o=$FS/outputs/cc_eval_$1_thinking_32k_both_vllm/summary.json
  [ -e "$o" ] && python3 -c "import json;d=json.load(open('$o'));x=d['metrics']['frontiercs']['reward'];a=d['metrics']['alebench']['performance'];print(f\"FCS={round(x.get('mean@5') if isinstance(x,dict) else x,2)}/ALE={round(a.get('mean@5'),0)}/n={d.get('complete_problem_count')}\")" 2>/dev/null || echo "-"; }
mls(){ local o=$FS/outputs/cc_mlsbench_cpu_${1}_devfix/summary.json
  [ -e "$o" ] && python3 -c "import json;print('MLS='+str(round(json.load(open('$o'))['mean_score'],4)))" 2>/dev/null || echo "-"; }
# re-submit a FCS/MLS eval iff it FAILED and its output is absent (port-strided script -> no re-collide)
recover_eval(){ # tag model_dir
  local tag="$1" dir="$2"
  if isfail "$(jstate cc-eval-$tag)" && [ ! -d "$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm" ]; then
    $PROC "$dir"/ 2>/dev/null
    sbatch -J cc-eval-$tag --time=06:00:00 --export=ALL,MODEL_PATH=$dir,TAG=$tag $FS/slurm/cc_eval_thinking_both_ailab.sh >/dev/null 2>&1 \
      && echo "  [recover] re-ran FCS eval $tag" >>$STATUS
  fi
  if isfail "$(jstate cc-mlsdevfix-$tag)" && [ ! -d "$FS/outputs/cc_mlsbench_cpu_${tag}_devfix" ]; then
    (cd $FS && sbatch -J cc-mlsdevfix-$tag --time=08:00:00 --export=ALL,MODEL_PATH=$dir,TAG=${tag}_devfix,MLSBENCH_ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev,EVAL_RESEARCHER_YEAR=2026 slurm/cc_eval_mlsbench_cpu_ailab.sh) >/dev/null 2>&1 \
      && echo "  [recover] re-ran MLS eval $tag" >>$STATUS
  fi
}

for i in $(seq 1 "$MAX_LOOPS"); do
  problem=""; echo "===== pipeline status $(date) (loop $i) =====" >$STATUS
  for V in $VARIANTS; do
    SFT=$ROOT/models_sft/sft_q35_a100_$V; SOUP=$ROOT/models_sft/soup_q35_a100_${V}_a${APCT}
    RLTAG=rl_${V}_a${APCT}; RLCK=$FS/checkpoints/rlm_amplify_v3/$RLTAG; HF=$ROOT/models/${RLTAG}_step40_hf
    # (1) submit matched RL-after eval once HF lands
    if have_full "$HF" && [ ! -d "$FS/outputs/cc_eval_rlafter_${RLTAG}_thinking_32k_both_vllm" ] && ! isfail "$(jstate cc-eval-rlafter_$RLTAG)"; then
      $PROC "$HF"/ 2>/dev/null
      sbatch -J cc-eval-rlafter_$RLTAG --time=06:00:00 --export=ALL,MODEL_PATH=$HF,TAG=rlafter_$RLTAG $FS/slurm/cc_eval_thinking_both_ailab.sh >/dev/null 2>&1
      (cd $FS && sbatch -J cc-mlsdevfix-rlafter_$RLTAG --time=08:00:00 --export=ALL,MODEL_PATH=$HF,TAG=rlafter_${RLTAG}_devfix,MLSBENCH_ROOT=/scratch/gpfs/CHIJ/bohan/MLS-Bench-dev,EVAL_RESEARCHER_YEAR=2026 slurm/cc_eval_mlsbench_cpu_ailab.sh) >/dev/null 2>&1
      echo "  [$V] submitted matched RL-after eval (HF ready)" >>$STATUS
    fi
    # (2) auto-recover failed evals (the recurring port-collision class)
    recover_eval "sftr2_$V"          "$SFT"
    recover_eval "soupr2_${V}_a$APCT" "$SOUP"
    recover_eval "rlafter_$RLTAG"     "$HF"
    # (3) status
    { echo "[$V]";
      echo "  SFT   $([ -e $SFT/config.json ] && echo done || echo .)   raw:  $(fcs sftr2_$V) $(mls sftr2_$V)"
      echo "  soup  $([ -e $SOUP/config.json ] && echo done || echo .)  a$APCT: $(fcs soupr2_${V}_a$APCT) $(mls soupr2_${V}_a$APCT)"
      echo "  RL    $([ -d $RLCK/global_step_40 ] && echo done || echo .)  after:$(fcs rlafter_$RLTAG) $(mls rlafter_$RLTAG)"
    } >>$STATUS
    # (4) real-stage failure detection (train/soup/RL). cc-rlexport only if HF ALSO missing.
    for jn in sft_$V soup_${V}_a$APCT rlm3_$RLTAG; do
      isfail "$(jstate $jn)" && { echo "  [$V] STAGE $jn FAILED" >>$STATUS; problem="yes"; }
    done
    isfail "$(jstate cc-rlexport-$RLTAG)" && ! have_full "$HF" && { echo "  [$V] STAGE export FAILED + no HF" >>$STATUS; problem="yes"; }
  done
  done_all="yes"; for V in $VARIANTS; do [ -e "$FS/outputs/cc_eval_rlafter_rl_${V}_a${APCT}_thinking_32k_both_vllm/summary.json" ] || done_all=""; done
  cat $STATUS
  [ -n "$problem" ] && { echo "WATCHDOG: real stage failure — exiting to alert."; exit 1; }
  [ -n "$done_all" ] && { echo "WATCHDOG: all pipelines reached RL-after eval — DONE."; exit 0; }
  sleep "$CHECK_S"
done
echo "WATCHDOG: max loops reached."
