#!/usr/bin/env bash
# Helper for rlv4_ckpt_eval_waiter.sh: export ONE FSDP checkpoint to HF, copy the
# tokenizer/processor files, then submit the four-bench eval chain for it.
# Env: CKPT_DIR (…/global_step_N/actor), HF_OUT, START_MODEL, TAG.
set -euo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
cd "$FS"

: "${CKPT_DIR:?}" "${HF_OUT:?}" "${START_MODEL:?}" "${TAG:?}"

if [ ! -e "$HF_OUT/config.json" ]; then
  echo "[export] $CKPT_DIR -> $HF_OUT"
  # The CPU partition's default python is miniconda's (transformers 4.46), which
  # does not know model_type qwen3_5 and fails AutoConfig with a ValueError.
  # Pin the venv interpreter that carries transformers 5.x.
  PY="${FS_EXPORT_PYTHON:-$FS/.venv-vllm023/bin/python}"
  [ -x "$PY" ] || PY=/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf/bin/python
  echo "[export] interpreter: $PY ($($PY -c 'import transformers;print(transformers.__version__)'))"
  "$PY" scripts/merge_fsdp_to_hf.py --ckpt "$CKPT_DIR" --output "$HF_OUT"
else
  echo "[export] $HF_OUT already present; skipping merge"
fi

# processor/tokenizer 四件套 (vLLM refuses to serve Qwen3.5 without these)
for f in tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json \
         merges.txt preprocessor_config.json chat_template.jinja generation_config.json; do
  [ -e "$START_MODEL/$f" ] && cp -n "$START_MODEL/$f" "$HF_OUT/" || true
done
ls -la "$HF_OUT" | head -15

# four-bench submission: never edit the shared submit script in place
TMP="/tmp/eval_${TAG}.sh"
cp scripts/cc_eval_rlsy_submit.sh "$TMP"
TAG="$TAG" HF_OUT="$HF_OUT" TMP="$TMP" python - <<'PY'
import os, re
p = os.environ["TMP"]
s = open(p).read()
row = 'MODELS=(\n  "%s|%s"\n)' % (os.environ["TAG"], os.environ["HF_OUT"])
s2, n = re.subn(r'MODELS=\(\n.*?\n\)', row, s, flags=re.S)
assert n == 1, f"MODELS block rewrite failed (n={n})"
open(p, "w").write(s2)
print("[submit] MODELS ->", row.replace("\n", " "))
PY
bash "$TMP" both
