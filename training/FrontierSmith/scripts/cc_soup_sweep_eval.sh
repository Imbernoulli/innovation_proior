#!/usr/bin/env bash
# Per-model model-soup alpha-sweep + auto-eval for 9B SFT models.
# For a trained SFT model, merge it with base at several alphas (soup = a*SFT+(1-a)*base)
# and submit FCS/ALE+research eval for each soup. Each model's best alpha differs, so we
# sweep. The raw-SFT eval is submitted separately (cc_eval_maintr3_9b_submit-style).
#   bash cc_soup_sweep_eval.sh <tag> <sft_model_dir> [alpha_csv]
# e.g. bash cc_soup_sweep_eval.sh maintr3_clean /path/sft_q35_clean_maintr3 "10,20,30"
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS="$ROOT/FrontierSmith"
BASE="$FS/models/Qwen3.5-9B-bf16"
cd "$FS"
TAG="${1:?need tag}"; SFT="${2:?need sft model dir}"; ALPHAS="${3:-10,20,30}"
EVAL_ONLY="${EVAL_ONLY:-0}"   # 1 = models already exist, just (re)submit evals, no merge
[ -e "$SFT/config.json" ] || { echo "ERROR: $SFT has no config.json"; exit 1; }
# Eval routing (command-line overrides the eval scripts' #SBATCH partition/gres). Default
# OFFLOAD so sweeps don't compete with ailab RL training: FCS/ALE -> gpu-ee (A100, chij);
# research -> pli (H100, goedelprover; A100 sm80 breaks the research evaluator). Set
# EV_DEST/RES_DEST="" to fall back to the scripts' native ailab target.
# FCS/ALE -> pli, NOT gpu-ee. gpu-ee has only TWO nodes, and each eval starts go-judge +
# apptainer + vLLM against node-local /tmp; a 3-alpha sweep is 6 shard jobs, which piled
# onto those 2 nodes and exhausted local disk -> go-judge died with
# "prefork environment failed ... fork/exec /proc/self/exe: no space left on device"
# (GPFS had 5.4P free at the time -- the exhausted filesystem is node-local, not /scratch).
# pli spreads the same jobs over ~30 nodes. Keep gpu-ee for one-off single evals.
EV_DEST="${EV_DEST:---partition=ailab --account=chij --gres=gpu:1}"
RES_DEST="${RES_DEST:---partition=pli --account=goedelprover --qos=pli-low --gres=gpu:1}"
MEMFLAG="--mem=110G"       # FCS/ALE evals (vLLM server): 110G is plenty
MEMFLAG_RES="--mem=160G"   # research evals run CPU evaluators -> need 160G (OOM'd at 110G; <180G/GPU soft limit)

# 9B eval helpers (2-way sharded FCS/ALE + research), reused from cc_eval_maintr3_9b_submit.
# Shard wall is 6h, NOT 3h: a 172-problem x 5-sample FCS/ALE shard takes ~2:15 on ailab
# H200 but ~3.7h on pli H100 and longer on the gpu-ee A100s this routes to by default, so
# a 3h wall TIMEOUTs at ~80% and burns a whole scheduling round-trip before the RESUME
# wave picks it up. (Raising an already-queued job instead of resubmitting:
# `scontrol update jobid=<id> TimeLimit=06:00:00` works while PENDING and keeps the queue
# position; it is denied once the job is RUNNING.)
submit_fcsale () { local tag="$1" mp="$2" dep="$3"
  local ob="$FS/outputs/cc_eval_${tag}_thinking_32k_both_vllm"
  local common="MODEL_PATH=$mp,TAG=$tag,NUM_SHARDS=2,MAX_NUM_SEQS=128,CONCURRENCY=96,REQUEST_TIMEOUT=3600"
  local s0 s1 s0w2 s1w2
  s0=$(sbatch --parsable $EV_DEST --time=06:00:00 $MEMFLAG ${dep:+--dependency=afterok:$dep} --job-name="ev-${tag}-s0" --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" slurm/cc_eval_thinking_both_ailab.sh)
  s1=$(sbatch --parsable $EV_DEST --time=06:00:00 $MEMFLAG ${dep:+--dependency=afterok:$dep} --job-name="ev-${tag}-s1" --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" slurm/cc_eval_thinking_both_ailab.sh)
  s0w2=$(sbatch --parsable $EV_DEST --time=02:00:00 $MEMFLAG --dependency=afterany:$s0 --job-name="ev-${tag}-s0w2" --export=ALL,$common,SHARD_IDX=0,OUTPUT_DIR="$ob/shard_0",SAMPLES_JSONL="$ob/shard_0/samples.jsonl",SUMMARY_JSON="$ob/shard_0/summary_shard.json" slurm/cc_eval_thinking_both_ailab.sh)
  s1w2=$(sbatch --parsable $EV_DEST --time=02:00:00 $MEMFLAG --dependency=afterany:$s1 --job-name="ev-${tag}-s1w2" --export=ALL,$common,SHARD_IDX=1,OUTPUT_DIR="$ob/shard_1",SAMPLES_JSONL="$ob/shard_1/samples.jsonl",SUMMARY_JSON="$ob/shard_1/summary_shard.json" slurm/cc_eval_thinking_both_ailab.sh)
  sbatch --parsable --dependency=afterany:$s0w2:$s1w2 --job-name="ev-${tag}-agg" --export=ALL,MODE=fcsale,OUTPUT_BASE="$ob",EXPECTED_SAMPLES=910,MAX_ERRORS=12 slurm/cc_eval_agg_shards.sh >/dev/null
  echo "  fcsale $tag -> s0=$s0 s1=$s1"
}
submit_research () { local tag="$1" mp="$2" dep="$3"
  local rb="$FS/outputs/cc_eval_${tag}_research_thinking_32k_vllm"
  local rc="MODEL_PATH=$mp,TAG=$tag,NUM_SHARDS=2,GPU_MEMORY_UTILIZATION=0.62"
  local r0 r1
  r0=$(sbatch --parsable $RES_DEST --time=04:00:00 $MEMFLAG_RES ${dep:+--dependency=afterok:$dep} --job-name="evr-${tag}-s0" --export=ALL,$rc,SHARD_IDX=0,OUTPUT_DIR="$rb/shard_0",SAMPLES_JSONL="$rb/shard_0/samples.jsonl",SUMMARY_JSON="$rb/shard_0/summary_shard.json" slurm/cc_eval_research_ailab.sh)
  r1=$(sbatch --parsable $RES_DEST --time=04:00:00 $MEMFLAG_RES ${dep:+--dependency=afterok:$dep} --job-name="evr-${tag}-s1" --export=ALL,$rc,SHARD_IDX=1,OUTPUT_DIR="$rb/shard_1",SAMPLES_JSONL="$rb/shard_1/samples.jsonl",SUMMARY_JSON="$rb/shard_1/summary_shard.json" slurm/cc_eval_research_ailab.sh)
  local r0w2 r1w2
  r0w2=$(sbatch --parsable $RES_DEST --time=02:30:00 $MEMFLAG_RES --dependency=afterany:$r0 --job-name="evr-${tag}-s0w2" --export=ALL,$rc,SHARD_IDX=0,OUTPUT_DIR="$rb/shard_0",SAMPLES_JSONL="$rb/shard_0/samples.jsonl",SUMMARY_JSON="$rb/shard_0/summary_shard.json" slurm/cc_eval_research_ailab.sh)
  r1w2=$(sbatch --parsable $RES_DEST --time=02:30:00 $MEMFLAG_RES --dependency=afterany:$r1 --job-name="evr-${tag}-s1w2" --export=ALL,$rc,SHARD_IDX=1,OUTPUT_DIR="$rb/shard_1",SAMPLES_JSONL="$rb/shard_1/samples.jsonl",SUMMARY_JSON="$rb/shard_1/summary_shard.json" slurm/cc_eval_research_ailab.sh)
  sbatch --parsable --dependency=afterany:$r0w2:$r1w2 --job-name="evr-${tag}-agg" --export=ALL,MODE=research,OUTPUT_BASE="$rb",EXPECTED_SAMPLES=320,MAX_ERRORS=20 slurm/cc_eval_agg_shards.sh >/dev/null
  echo "  research $tag -> r0=$r0 r1=$r1"
}

IFS=',' read -ra AS <<<"$ALPHAS"
for a in "${AS[@]}"; do
  af=$(python3 -c "print($a/100)")
  stag="${TAG}_a${a}"
  sout="$ROOT/models_sft/soup_q35_${stag}"
  mjid=""
  if [ -e "$sout/config.json" ]; then
    echo "[soup exists] $stag"
  elif [ "$EVAL_ONLY" = "1" ]; then
    echo "[skip $stag: EVAL_ONLY but model missing]"; continue
  else
    # Merge on the cpu partition (memory-mapped, single-thread-safe). NOTE: the gpu and
    # gputest partitions REJECT direct submission for our accounts (goedelprover=pli /
    # chij has no gpu-partition grant), so CPU is the only self-service merge path; if a
    # working gputest submit line exists, swap PARTITION/account/qos and re-add --device cuda.
    mjid=$(sbatch --parsable --partition=cpu --cpus-per-task=8 --mem=96G --time=00:50:00 \
      --job-name="soup-${stag}" --output="$FS/logs/%x-%j.out" --error="$FS/logs/%x-%j.err" \
      --wrap="cd $FS && OMP_NUM_THREADS=8 $ROOT/envs/sft_lf/bin/python scripts/cc_model_soup_merge.py --sft '$SFT' --base '$BASE' --alpha $af --out '$sout' --device cpu")
    echo "[soup submit] $stag alpha=$af jid=$mjid"
  fi
  submit_fcsale  "soup_${stag}" "$sout" "$mjid"
  submit_research "soup_${stag}" "$sout" "$mjid"
done
echo "done soup-sweep+eval for $TAG (alphas=$ALPHAS)"
