#!/usr/bin/env bash
# Four-bench eval for the finished rlv12 arms, sized to actually SATURATE ailab.
#
# ailab caps a user at 16 GPUs. The way to use all of them is many 1-GPU jobs,
# not a few multi-GPU ones: 4 arms x (2 fcsale shards + 2 research shards) = 16
# concurrent single-GPU jobs, exactly filling the cap.
#
# CPU sizing is the real constraint, and ailab ENFORCES a ratio: "Only 8 cores per
# GPU" -- asking 16 cores on 1 GPU is rejected at submit time, not silently capped.
# That matters because FCS/ALE scoring compiles and runs C++ alongside generation;
# at 8 cores a node ran load 12.8 and the GPU starved. So to get 16 cores we must
# take 2 GPUs. 8 jobs x 2 GPUs = 16, still exactly the per-user cap, with double
# the CPU for the decoupled scorer pool. (vLLM uses one GPU; the second buys cores.)
#
# MLS is CPU-side and submitted separately (slurm/cc_eval_mlsbench_cpu_ailab.sh).
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"
MR=/scratch/gpfs/CHIJ/bohan/fs/models_rl

ARMS=("${@:-base loraIM soupNEW10 soupWD03_20}")
read -r -a ARMS <<< "${ARMS[*]}"

AILAB=(--partition=ailab --qos=gpu-short --gres=gpu:2 -c 16 --mem=220G)

for arm in "${ARMS[@]}"; do
  tag="rlv12_${arm}_s20"
  mp="$MR/${tag}_hf"
  if [ ! -f "$mp/config.json" ]; then
    echo "SKIP $tag: not exported yet ($mp)"
    continue
  fi
  tmp="/tmp/rlv12_${arm}.sh"
  cp scripts/cc_eval_rlsy_submit.sh "$tmp"
  TAG="$tag" MP="$mp" TMP="$tmp" python3 - <<'PY'
import os, re
p = os.environ["TMP"]; s = open(p).read()
row = 'MODELS=(\n  "%s|%s"\n)' % (os.environ["TAG"], os.environ["MP"])
s2, n = re.subn(r'MODELS=\(\n.*?\n\)', row, s, flags=re.S)
assert n == 1, "MODELS rewrite failed"
open(p, "w").write(s2)
PY
  echo "== $tag"
  FS_EVAL_PART="${AILAB[*]}" FS_RES_PART="${AILAB[*]}" EVAL_DECOUPLE=1 \
    bash "$tmp" both 2>&1 | grep -E "^fcsale|^research|SKIP"
done
echo "submitted. ailab cap is 16 GPUs; 4 arms x 4 shards fills it exactly."
