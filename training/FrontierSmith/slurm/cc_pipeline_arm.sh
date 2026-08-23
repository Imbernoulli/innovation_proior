#!/usr/bin/env bash
# One arm of the SFT pipeline, self-chaining: TRAIN -> (soup | lora-merge) -> EVAL.
#
# Why chained rather than three manual rounds: every hand-off was previously a
# human step, and each missed hand-off cost a full queue wait (the rlv13 HF
# exports sat done-but-unevaluated for two days for exactly this reason). Each
# stage submits the next with --dependency=afterok, so a whole arm walks from a
# training config to four-board numbers with no supervision.
#
# Submit:
#   ARM=<name> CFG=<lf yaml> KIND=full|lora ALPHAS="0.1 0.2 0.3" \
#     sbatch slurm/cc_pipeline_arm.sh
#
# Stages read only what the previous stage wrote; a failed stage stops the chain
# (afterok), leaving the partial products on disk for inspection.
#SBATCH --job-name=cc-pipe
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -uo pipefail
ROOT=/scratch/gpfs/CHIJ/bohan/fs
FS=$ROOT/FrontierSmith
LF=$ROOT/LF-innov
MS=$ROOT/models_sft
: "${ARM:?set ARM}" "${CFG:?set CFG}"
KIND="${KIND:-full}"
ALPHAS="${ALPHAS:-0.1 0.2 0.3}"
EVAL_ALPHAS="${EVAL_ALPHAS:-$ALPHAS}"

# ---- stage 1: train --------------------------------------------------------
cd "$LF"
TRAIN=$(sbatch --parsable --job-name="p1t-$ARM" --partition=ailab \
  --gres=gpu:4 --cpus-per-task=32 --mem=400G --time=16:00:00 \
  cc-sft-innov.sh "$CFG")
echo "[pipe:$ARM] train -> $TRAIN"

# ---- stage 2: soup (full-FT) or scaled merge (LoRA) ------------------------
if [ "$KIND" = "lora" ]; then
  MERGE=$(sbatch --parsable --job-name="p2m-$ARM" --partition=ailab \
    --gres=gpu:1 --cpus-per-task=8 --mem=250G --time=03:00:00 \
    --dependency=afterok:$TRAIN \
    --export=ALL,ARMS="$ARM",SCALES="$ALPHAS" \
    "$FS/slurm/cc_pipe_merge_lora.sh")
  SUFFIXES=""; for a in $EVAL_ALPHAS; do SUFFIXES="$SUFFIXES _s${a/0./0}_merged"; done
else
  MERGE=$(sbatch --parsable --job-name="p2s-$ARM" --partition=ailab \
    --gres=gpu:1 --cpus-per-task=8 --mem=250G --time=03:00:00 \
    --dependency=afterok:$TRAIN \
    --export=ALL,ARMS="$ARM",ALPHAS="$ALPHAS" \
    "$FS/slurm/cc_pipe_soup.sh")
  SUFFIXES=""; for a in $EVAL_ALPHAS; do SUFFIXES="$SUFFIXES _soup${a/0./0}"; done
fi
echo "[pipe:$ARM] merge -> $MERGE (suffixes:$SUFFIXES)"

# ---- stage 3: eval each merged product ------------------------------------
# Split architecture (user ruling): the GPU job ONLY serves; generation requests
# and judging run as cpu-partition clients. cc_eval_split_submit.sh owns that
# fan-out, so the pipeline hands off to it rather than re-implementing it. The
# submitter itself needs no GPU, but it must not run until the merge exists --
# hence the tiny dependent shim job below.
cd "$FS"
for suf in $SUFFIXES; do
  M="$MS/sft_q35_${ARM}${suf}"
  T="sft_q35_${ARM}${suf}"
  EV=$(sbatch --parsable --job-name="p3s-${ARM}${suf}" --partition=cpu \
    --cpus-per-task=1 --mem=2G --time=00:10:00 \
    --dependency=afterok:$MERGE \
    --wrap="cd $FS && EVAL_SYS_PROMPT_MODE=bare bash slurm/cc_eval_split_submit.sh '$M' '$T' both 2")
  echo "[pipe:$ARM] eval-split ${ARM}${suf} -> submitter $EV"
done
echo "[pipe:$ARM] chain submitted"
