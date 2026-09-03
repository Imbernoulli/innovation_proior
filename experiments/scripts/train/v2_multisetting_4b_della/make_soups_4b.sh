#!/usr/bin/env bash
#SBATCH --job-name=soup4b
#SBATCH --partition=cpu
#SBATCH --account=chij
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# make_soups_4b.sh <arm> <alpha>  -- 4B twin of make_soups.sh.
#   full-FT arm : soup = alpha*sft + (1-alpha)*base   (cc_model_soup_merge.py)
#   LoRA arm    : soup = base + alpha*dW              (soup_lora.py, identical maths)
# Same maths, same scripts; only BASE and the model dir differ. 80G of RAM is enough
# here (the 9B twin asks 120G) because the 4B state dict is 9.3G, not 36G.
set -euo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
BASE=$D/models/Qwen3.5-4B
PY=/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf/bin/python
ARM="${1:?arm}"; A="${2:?alpha}"
TAG=$(printf '%s_soup%02d' "$ARM" "$(python3 -c "print(round($A*100))")")
OUT="$D/models/$TAG"
export HF_HOME="$D/.hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f "$OUT/config.json" ] && { echo "$OUT exists, skip"; exit 0; }

if [ -f "$D/models/$ARM/adapter_config.json" ]; then
  "$PY" "$D/scripts/soup_lora.py" --adapter "$D/models/$ARM" --base "$BASE" --alpha "$A" --out "$OUT"
else
  "$PY" /scratch/gpfs/CHIJ/bohan_1/innovation_proior/experiments/scripts/train/cc_model_soup_merge.py \
      --sft "$D/models/$ARM" --base "$BASE" --alpha "$A" --out "$OUT"
fi
echo "DONE $OUT"
ls -la "$OUT"
