#!/usr/bin/env bash
# =============================================================================
# 35B RL smoke checker + auto-advance (runs afterany the smoke job).
#
#  - Verifies SUCCESS: latest ckpt step >= 2 AND global_step_N/actor holds all
#    8 model_world_size_8_rank_*.pt shards totalling >= 55 GiB (complete
#    model-only save; a save-OOM leaves a truncated folder -- 9B lesson).
#  - Extracts real step timings + max-memory lines into the pipeline status log.
#  - PASS  -> submits the FORMAL 20-step runs (r32s01 + base control, each with
#             w2/w3 continuation windows) using the knobs of the level that
#             passed, plus one ckpt-eval waiter per arm.
#  - FAIL  -> submits the next smoke in the OOM backoff cascade + its checker:
#             L1 TP=1 GMU=0.65 RESP=32768 | L2 TP=1 GMU=0.55 RESP=32768
#             L3 TP=1 GMU=0.55 RESP=24576 | L4 TP=2 GMU=0.35 RESP=24576
#             L4 fail -> loud ALARM in the status log, human needed.
#
# Env: LEVEL (1..4), EXP (experiment name of the smoke this job checks).
# =============================================================================
#SBATCH --job-name=rl35b_check
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
set -uo pipefail

ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
cd "$FS"
LEVEL="${LEVEL:-1}"
EXP="${EXP:-rl35b_r32s01_smoke}"
CK="$FS/checkpoints/rl_frontiersmith_synth/$EXP"
STATUS="$FS/logs/rl35b_pipeline_status.log"
log() { echo "[$(date '+%F %T')] [check-L$LEVEL] $*" | tee -a "$STATUS"; }

# ---- backoff table ----------------------------------------------------------
lvl_knobs() {  # level -> "TP GMU MAXRESP MAXLEN"
  case "$1" in
    1) echo "1 0.65 32768 45056" ;;
    2) echo "1 0.55 32768 45056" ;;
    3) echo "1 0.55 24576 35840" ;;
    4) echo "2 0.35 24576 35840" ;;
    *) echo "" ;;
  esac
}

# ---- disk guard ---------------------------------------------------------------
freeg=$(df -BG --output=avail /scratch/gpfs/CHIJ | tail -1 | tr -dc 0-9)
if [ "${freeg:-0}" -lt 500 ]; then
  log "ALARM: disk ${freeg}G < 500G -- STOP, not submitting anything. Human needed."
  exit 9
fi

# ---- verify smoke -------------------------------------------------------------
ok=0
step=$(cat "$CK/latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc 0-9 || true)
if [ -n "${step:-}" ] && [ "$step" -ge 2 ]; then
  A="$CK/global_step_${step}/actor"
  nsh=$(ls "$A"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
  szg=$(du -s --apparent-size --block-size=1G "$CK/global_step_${step}" 2>/dev/null | cut -f1)
  if [ "$nsh" -eq 8 ] && [ "${szg:-0}" -ge 55 ]; then ok=1; fi
  log "smoke $EXP: step=$step shards=$nsh size=${szg:-?}G -> $([ $ok = 1 ] && echo PASS || echo INCOMPLETE-CKPT)"
else
  log "smoke $EXP: no checkpoint >= step 2 (latest='${step:-none}') -> FAIL"
fi

# ---- timing report ------------------------------------------------------------
lg=$(ls -t "$FS"/logs/${EXP}-*.out 2>/dev/null | head -1)
if [ -n "${lg:-}" ]; then
  log "timings from $(basename "$lg"):"
  grep -oE "timing_s/(gen|update_actor|update_weights|step):[0-9.]+" "$lg" | tail -16 | tee -a "$STATUS" || true
  grep -oE "perf/max_memory_[a-z]+_gb:[0-9.]+" "$lg" | tail -6 | tee -a "$STATUS" || true
  grep -E "CUDA out of memory|OutOfMemory|Traceback" "$lg" | tail -3 | tee -a "$STATUS" || true
fi

if [ "$ok" = 1 ]; then
  read -r TP GMU MAXRESP MAXLEN <<< "$(lvl_knobs "$LEVEL")"
  log "PASS at L$LEVEL (TP=$TP GMU=$GMU RESP=$MAXRESP) -> submitting FORMAL 20-step (r32s01 + base) + eval waiters"
  out=$(SMOKE=0 STEPS=20 SAVE=5 TB=32 RN=8 MB=16 KEEP=10 WALL=23:59:00 \
        TP=$TP GMU=$GMU MAXRESP=$MAXRESP MAXLEN=$MAXLEN \
        ONLY="r32s01 base" bash "$FS/scripts/cc_rl35b_synth_submit.sh")
  echo "$out" | tee -a "$STATUS"
  for arm in r32s01 base; do
    case "$arm" in
      r32s01) sm="$ROOT/models_sft/lora_q36_35bA3b_clean_nom_r32_s01_merged" ;;
      base)   sm="$ROOT/models/Qwen3.6-35B-A3B" ;;
    esac
    wj=$(sbatch --parsable --job-name="rl35b_evalwaiter_${arm}" \
      --export=ALL,EXP="rl35b_${arm}",ARMTAG="$arm",START_MODEL="$sm" \
      "$FS/scripts/rl35b_ckpt_eval_waiter.sh")
    log "eval waiter for rl35b_${arm}: job $wj"
  done
  exit 0
fi

# ---- FAIL: distinguish HOST-RAM OOM from GPU OOM ------------------------------
# A slurm oom_kill WITHOUT a CUDA OOM means the job blew its HOST memory cgroup
# (the smoke-11169862 failure mode: Adam offload 268G + ref fp32 on CPU 140G +
# Ray object store). GPU backoff levels cannot fix that -- since the submit
# helper now already requests the full node (1450G) for 8-GPU runs, a host OOM
# at 1450G needs a human (shrink Ray object store / ref handling), not a cascade.
errlog=$(ls -t "$FS"/logs/${EXP}-*.err 2>/dev/null | head -1)
if [ -n "${errlog:-}" ] && grep -q "oom_kill event" "$errlog" 2>/dev/null \
   && ! grep -qE "CUDA out of memory|OutOfMemoryError" "$errlog" "$lg" 2>/dev/null; then
  # Was this smoke already run with full-node memory? Check the sbatch record.
  memreq=$(sacct -n -X --format=ReqMem -j "$(basename "$errlog" | grep -oE '[0-9]+' | tail -1)" 2>/dev/null | tr -dc '0-9')
  if [ "${memreq:-0}" -ge 1400 ] 2>/dev/null; then
    log "ALARM: HOST-RAM OOM even at full-node memory (${memreq}G) -- GPU backoff will not help. Human needed."
    exit 1
  fi
  log "HOST-RAM OOM at ${memreq:-?}G (not GPU) -> resubmitting SAME level L$LEVEL with full-node 1450G memory"
  read -r TP GMU MAXRESP MAXLEN <<< "$(lvl_knobs "$LEVEL")"
  out=$(SMOKE=1 STEPS=3 SAVE=2 TB=8 RN=4 MB=8 WALL=08:00:00 MEM=1450 \
        TP=$TP GMU=$GMU MAXRESP=$MAXRESP MAXLEN=$MAXLEN \
        EXPSUFFIX="_m$LEVEL" ONLY=r32s01 FRESH_START=1 \
        bash "$FS/scripts/cc_rl35b_synth_submit.sh")
  echo "$out" | tee -a "$STATUS"
  sj=$(echo "$out" | grep -oE '\-> [0-9]+' | grep -oE '[0-9]+' | head -1)
  cj=$(sbatch --parsable --dependency=afterany:$sj --job-name="rl35b_check_L${LEVEL}m" \
    --export=ALL,LEVEL=$LEVEL,EXP="rl35b_r32s01_m${LEVEL}_smoke" \
    "$FS/scripts/rl35b_smoke_check_and_advance.sh")
  log "mem-fixed smoke=$sj checker=$cj"
  exit 0
fi

# ---- FAIL: back off -----------------------------------------------------------
NEXT=$((LEVEL + 1))
knobs=$(lvl_knobs "$NEXT")
if [ -z "$knobs" ]; then
  log "ALARM: smoke failed at final backoff level L$LEVEL -- cascade exhausted. Human needed."
  exit 1
fi
read -r TP GMU MAXRESP MAXLEN <<< "$knobs"
log "FAIL at L$LEVEL -> backing off to L$NEXT (TP=$TP GMU=$GMU RESP=$MAXRESP LEN=$MAXLEN)"
out=$(SMOKE=1 STEPS=3 SAVE=2 TB=8 RN=4 MB=8 WALL=08:00:00 \
      TP=$TP GMU=$GMU MAXRESP=$MAXRESP MAXLEN=$MAXLEN \
      EXPSUFFIX="_b$NEXT" ONLY=r32s01 FRESH_START=1 \
      bash "$FS/scripts/cc_rl35b_synth_submit.sh")
echo "$out" | tee -a "$STATUS"
sj=$(echo "$out" | grep -oE '\-> [0-9]+' | grep -oE '[0-9]+' | head -1)
if [ -n "${sj:-}" ]; then
  cj=$(sbatch --parsable --dependency=afterany:$sj --job-name="rl35b_check_L$NEXT" \
    --export=ALL,LEVEL=$NEXT,EXP="rl35b_r32s01_b${NEXT}_smoke" \
    "$FS/scripts/rl35b_smoke_check_and_advance.sh")
  log "next smoke=$sj checker=$cj"
else
  log "ALARM: could not parse next smoke job id -- cascade broken. Human needed."
  exit 1
fi
