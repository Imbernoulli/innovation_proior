#!/usr/bin/env bash
# ONE COMMAND: ship a model from gpublaze to jiaolab and start its eval there.
# *** RUN THIS ON gpublaze *** (it rsyncs out and then ssh's in).
#
#   bash scripts/jiaolab/eval_model_from_gpublaze.sh <MODEL_DIR> [TAG] [KIND]
#
#   MODEL_DIR  absolute path, or a name relative to $MODEL_ROOT
#              (default /srv/home/bohanlyu/models_sft), e.g.
#                v2_multisetting_4b/full_wd01
#                agentic_ablation_4b/soup_withag_a10
#   TAG        served-model name + output/log prefix
#              (default: ja_<basename of the parent>_<basename>, sanitized)
#   KIND       fcsale (default) | ale | fcs
#
# What it does:
#   1. rsync (nice+ionice, so gpublaze training is unaffected) ONLY the serving
#      payload -- config/tokenizer/*.safetensors -- excluding checkpoint-*/,
#      optimizer state, training_args.bin and plots. ~9G for a 4B bf16 model
#      instead of ~26G for the whole training dir.
#   2. ssh jiaolab -> scripts/jiaolab/launch_pool_eval.sh: 2 auto-picked free
#      A100s, one TP=1 engine each, 2 pinned clients, full protocol
#      (n=5 / 32768 / temp 1.0 / top_p 0.95 / top_k 20 / presence 1.5 / y2026).
#   3. Print the log + output locations and return immediately (the eval runs
#      detached under setsid on jiaolab).
#
# Env knobs: MODEL_ROOT, JIAOLAB_HOST (default "jiaolab"),
#            JIAOLAB_MODELS (default /home/bohan/models_from_gpublaze),
#            GPUA/GPUB (skip auto-pick), DRY_RUN=1.
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/srv/home/bohanlyu/models_sft}"
JHOST="${JIAOLAB_HOST:-jiaolab}"
JMODELS="${JIAOLAB_MODELS:-/home/bohan/models_from_gpublaze}"
JFS="${JIAOLAB_FS_ROOT:-/home/bohan/innovation_proior/training/FrontierSmith}"

SRC_ARG="${1:?usage: eval_model_from_gpublaze.sh <MODEL_DIR|name> [TAG] [KIND]}"
case "$SRC_ARG" in
  /*) SRC="$SRC_ARG" ;;
  *)  SRC="$MODEL_ROOT/$SRC_ARG" ;;
esac
SRC="${SRC%/}"
[ -d "$SRC" ] || { echo "ERROR: model dir not found: $SRC" >&2; exit 1; }
[ -f "$SRC/config.json" ] || { echo "ERROR: $SRC has no config.json -- not a servable HF model dir" >&2; exit 1; }

DEFAULT_TAG="ja_$(basename "$(dirname "$SRC")")_$(basename "$SRC")"
TAG="${2:-$DEFAULT_TAG}"
TAG="$(printf '%s' "$TAG" | tr -c 'A-Za-z0-9_.-' '_')"
KIND="${3:-fcsale}"
DEST="$JMODELS/$(basename "$(dirname "$SRC")")__$(basename "$SRC")"

echo "[ship] src   = $SRC"
echo "[ship] dest  = $JHOST:$DEST"
echo "[ship] tag   = $TAG   kind = $KIND"
[ "${DRY_RUN:-0}" = "1" ] && { echo "[ship] DRY_RUN=1, stopping here"; exit 0; }

# Disk guard: jiaolab's / is ~93% full. Refuse to ship if the payload would not
# leave a comfortable margin (models can be deleted after their eval finishes:
#   ssh jiaolab "rm -rf $DEST").
NEED_GB=$(du -sBG --exclude='checkpoint-*' "$SRC" | cut -f1 | tr -d 'G')
FREE_GB=$(ssh "$JHOST" "df -BG --output=avail / | tail -1" | tr -dc '0-9')
echo "[ship] payload ~${NEED_GB}G, jiaolab free ${FREE_GB}G"
if [ "$FREE_GB" -lt $((NEED_GB + 40)) ]; then
  echo "ERROR: jiaolab has only ${FREE_GB}G free for a ~${NEED_GB}G model (want payload+40G)." >&2
  echo "  Delete a finished model under $JMODELS first." >&2
  exit 1
fi

ssh "$JHOST" "mkdir -p '$DEST'"
# Serving payload only. --exclude order matters: checkpoint dirs first.
nice -n 19 ionice -c3 rsync -a --info=progress2 \
  --exclude 'checkpoint-*/' --exclude 'optimizer*' --exclude 'scheduler*' \
  --exclude 'rng_state*' --exclude 'training_args.bin' --exclude '*.png' \
  --exclude 'trainer_log.jsonl' --exclude 'wandb/' \
  "$SRC"/ "$JHOST:$DEST"/
echo "[ship] transfer done"

ssh "$JHOST" "ls '$DEST'/config.json >/dev/null" || { echo "ERROR: config.json missing at destination" >&2; exit 1; }

GPUARGS=""
if [ -n "${GPUA:-}" ] && [ -n "${GPUB:-}" ]; then GPUARGS=" $GPUA $GPUB"; fi
echo "[eval] launching pool eval on $JHOST ..."
# shellcheck disable=SC2029
ssh "$JHOST" "cd '$JFS' && setsid nohup bash scripts/jiaolab/launch_pool_eval.sh '$DEST' '$TAG' '$KIND'$GPUARGS > logs/pool_launch_${TAG}.log 2>&1 < /dev/null & sleep 3; cat logs/pool_launch_${TAG}.log"

cat <<EOM

[eval] started. On $JHOST:
  launcher log : $JFS/logs/pool_launch_${TAG}.log
  serve logs   : $JFS/logs/serve_pool_${TAG}_gpu*.log
  client logs  : $JFS/logs/cli_${TAG}_*_pool.log
  outputs      : $JFS/outputs/cc_eval_${TAG}_thinking_32k_both_vllm/shard_*/
  progress     : ssh $JHOST "tail -f $JFS/logs/cli_${TAG}_both0_pool.log"
  aggregate    : ssh $JHOST "cd $JFS && .venv/bin/python scripts/reaggregate_all_summary.py"
EOM
