#!/usr/bin/env bash
# Spread pending FCS/ALE evals across EVERY lane we have access to, instead of
# letting them all queue behind one congested partition.
#
# Why: 60 eval jobs sat on the `gpu` partition with a ~17 h start estimate while
# pli (38 nodes x h100:8) and gpu-ee sat idle. Access (from sacctmgr assoc):
#   goedelprover -> pli-low      (pli partition, by far the most capacity)
#   chij         -> della-gpuee, gpu-medium (20 GPU), gpu-short (44 GPU), gpu-long
#
# All lanes run the SAME decoupled driver (EVAL_DECOUPLE defaults to 1 in
# scripts/eval_base_model_qwen35_9b_vllm_request.sh, which is read at RUNTIME),
# so lane choice affects only scheduling latency, never the scores: a record is
# a function of (model, prompt, sampling params, seed) and none of those vary by
# GPU model. Mixing A100 and H100 across arms is therefore safe for comparison.
#
# Usage: bash scripts/spread_evals.sh <tag> [tag...]
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"

# Lanes, cheapest-to-schedule first. Round-robined across tags.
LANES=(
  "--partition=pli --account=goedelprover --qos=pli-low --gres=gpu:h100:1"
  "--partition=gpu-ee --gres=gpu:a100:1"
  "--qos=gpu-medium --gres=gpu:a100:1"
  "--partition=pli --account=goedelprover --qos=pli-low --gres=gpu:h100:1"
  "--qos=gpu-short --gres=gpu:a100:1"
)

MR=/scratch/gpfs/CHIJ/bohan/fs/models_rl          # rlv10_* live here
MRI=$FS/models_rl                                  # rlsy_* live HERE (different dir!)
MSF=/scratch/gpfs/CHIJ/bohan/fs/models_sft
M9=$FS/models/Qwen3.5-9B-bf16

model_for () {  # tag -> path
  case "$1" in
    anchor_base)          echo "$M9" ;;
    anchor_soupNEW10)     echo "$MSF/soup_q35_innnew_ft_a10" ;;
    anchor_loraIM)        echo "$MSF/lora_q35_im_r32_s01_merged" ;;
    anchor_soupWD03_20)   echo "$MSF/soup_q35_innnew_wd03_ft_a20" ;;
    rlsy*)                echo "$MRI/$(echo "$1" | sed 's/^rlsyb/rlsy_base/;s/^rlsyL/rlsy_loraIM/;s/^rlsyS/rlsy_soupNEW10/;s/^rlsyW/rlsy_soupWD03_20/')_hf" ;;
    *)                    echo "$MR/${1}_hf" ;;
  esac
}

i=0
for tag in "$@"; do
  mp=$(model_for "$tag")
  if [ ! -f "$mp/config.json" ]; then
    echo "SKIP $tag: no config.json at $mp"
    continue
  fi
  # Cancel this tag's existing jobs first -- two jobs appending the same
  # samples.jsonl from different nodes corrupts it (learned the hard way).
  ids=$(squeue -u "$USER" -h -o "%i|%j" | awk -F'|' -v t="$tag" '$2 ~ ("^cc-eval-9b-" t "-") {print $1}')
  if [ -n "$ids" ]; then
    scancel $ids 2>/dev/null
    sleep 2
  fi

  lane="${LANES[$((i % ${#LANES[@]}))]}"
  i=$((i + 1))
  tmp="/tmp/spread_${tag}.sh"
  cp scripts/cc_eval_rlsy_submit.sh "$tmp"
  TAG="$tag" MP="$mp" TMP="$tmp" python3 - <<'PY'
import os, re
p = os.environ["TMP"]; s = open(p).read()
row = 'MODELS=(\n  "%s|%s"\n)' % (os.environ["TAG"], os.environ["MP"])
s2, n = re.subn(r'MODELS=\(\n.*?\n\)', row, s, flags=re.S)
assert n == 1, "MODELS rewrite failed"
open(p, "w").write(s2)
PY
  echo "== $tag -> [$lane]"
  FS_EVAL_PART="$lane" EVAL_DECOUPLE=1 bash "$tmp" fcsale 2>&1 | grep -E "^fcsale|SKIP"
done
echo "spread done."
