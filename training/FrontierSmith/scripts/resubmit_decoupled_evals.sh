#!/usr/bin/env bash
# Resubmit the full eval set with the DECOUPLED driver (EVAL_DECOUPLE=1 is the
# default in the launchers now). Resume is on, so samples already banked by the
# old coupled jobs are kept -- this only tops up the remainder.
#
# Routing is staggered to respect QOS caps: gpu-medium (24/user), gpu-test (3),
# pli (~6). FCS/ALE -> gpu-medium; Research -> pli. The decoupled smoke showed
# 2x throughput and +57% per-request decode (19.4 -> 30.4 tok/s/req).
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"
MR=/scratch/gpfs/CHIJ/bohan/fs/models_rl
MSF=/scratch/gpfs/CHIJ/bohan/fs/models_sft
M9=$FS/models/Qwen3.5-9B-bf16

export EVAL_DECOUPLE=1

# tag|model_path   (only models with a config.json)
FCSALE_MODELS=(
  "rlv10_base_s5|$MR/rlv10_base_s5_hf"
  "rlv10_base_s10|$MR/rlv10_base_s10_hf"
  "rlv10_base_s15|$MR/rlv10_base_s15_hf"
  "rlv10_base_s20|$MR/rlv10_base_s20_hf"
  "rlv10_soupNEW10_s5|$MR/rlv10_soupNEW10_s5_hf"
  "rlv10_soupNEW10_s10|$MR/rlv10_soupNEW10_s10_hf"
  "rlv10_soupWD03_20_s5|$MR/rlv10_soupWD03_20_s5_hf"
  "rlv10_soupWD03_20_s10|$MR/rlv10_soupWD03_20_s10_hf"
  "rlv10_soupWD03_20_s15|$MR/rlv10_soupWD03_20_s15_hf"
  "rlv10_loraIM_s5|$MR/rlv10_loraIM_s5_hf"
  "anchor_base|$M9"
  "anchor_soupNEW10|$MSF/soup_q35_innnew_ft_a10"
  "anchor_loraIM|$MSF/lora_q35_im_r32_s01_merged"
  "anchor_soupWD03_20|$MSF/soup_q35_innnew_wd03_ft_a20"
)

submit_one () {  # kind tag model
  local kind="$1" tag="$2" mp="$3"
  [ -f "$mp/config.json" ] || { echo "  SKIP $tag (no config)"; return; }
  local tmp="/tmp/eval_rs_${tag}.sh"
  cp scripts/cc_eval_rlsy_submit.sh "$tmp"
  TAG="$tag" MP="$mp" TMP="$tmp" python3 - <<'PY'
import os, re
p=os.environ["TMP"]; s=open(p).read()
row='MODELS=(\n  "%s|%s"\n)' % (os.environ["TAG"], os.environ["MP"])
s2,n=re.subn(r'MODELS=\(\n.*?\n\)', row, s, flags=re.S)
assert n==1; open(p,"w").write(s2)
PY
  bash "$tmp" "$kind"
}

KIND="${1:-both}"
echo "=== resubmitting decoupled evals ($KIND) ==="
for row in "${FCSALE_MODELS[@]}"; do
  IFS='|' read -r tag mp <<<"$row"
  submit_one "$KIND" "$tag" "$mp"
done
echo "done."
