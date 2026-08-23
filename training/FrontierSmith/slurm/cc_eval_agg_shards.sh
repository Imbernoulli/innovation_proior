#!/usr/bin/env bash
# =============================================================================
# Aggregate a 2-way sharded eval (NUM_SHARDS=2, SHARD_IDX=0/1) into the
# canonical top-level summary.json that single-shard runs produce.
#
# How: concatenate shard_0/samples.jsonl + shard_1/samples.jsonl (the driver's
# _load_existing dedups by (data_source, ground_truth, sample_idx) LAST-WINS,
# so stale error records are superseded by later good ones), gate on
# completeness, then re-invoke the SAME eval driver in --resume mode over the
# merged file: all planned samples are complete -> empty task list -> no vLLM /
# judge needed -> _summarize writes the full canonical summary.json.
# (<=MAX_ERRORS residual error records get a fast failed-regeneration attempt
# against a dead endpoint and stay conservatively scored 0 -- same convention
# as a single-shard run's tolerated judge timeouts.)
#
# Env:
#   MODE=fcsale|research   (default fcsale)
#   OUTPUT_BASE            (required: .../outputs/cc_eval_<TAG>_...vllm)
#   EXPECTED_SAMPLES       (910 for full FCS/ALE, 320 for research-full)
#   MAX_ERRORS             (12 fcsale / 20 research)
#   RESEARCH_DATA          (research mode; default full research.parquet)
# =============================================================================
#SBATCH --job-name=cc-eval-agg-shards
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith"
cd "$PROJECT_ROOT"

MODE="${MODE:-fcsale}"
: "${OUTPUT_BASE:?OUTPUT_BASE required}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-910}"
MAX_ERRORS="${MAX_ERRORS:-12}"

S0="$OUTPUT_BASE/shard_0/samples.jsonl"
S1="$OUTPUT_BASE/shard_1/samples.jsonl"
MERGED_DIR="$OUTPUT_BASE/merged"
MERGED="$MERGED_DIR/samples.jsonl"
mkdir -p "$MERGED_DIR"

for f in "$S0" "$S1"; do
  [ -s "$f" ] || { echo "ABORT: missing/empty shard file $f" >&2; exit 3; }
done
# Order matters: shard_0 first, shard_1 second -> last-wins dedup lets fresh
# shard_1 records supersede any stale (pre-shard) error records in shard_0.
cat "$S0" "$S1" > "$MERGED"

# Completeness gate BEFORE invoking the driver: never let the summary step run
# against a grossly incomplete merge (it would hammer the dead endpoint).
python3 - "$MERGED" "$EXPECTED_SAMPLES" "$MAX_ERRORS" <<'PY'
import json, sys
path, expected, max_err = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
recs = {}
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
        key = (str(r["data_source"]), str(r["ground_truth"]), int(r["sample_idx"]))
    except Exception:
        continue
    recs[key] = r  # last-wins
n = len(recs)
errs = sum(1 for r in recs.values() if r.get("error"))
print(f"[agg] merged: {n} distinct samples, {errs} errors (expected {expected}, max_errors {max_err})")
if n < expected:
    print(f"[agg] ABORT: only {n}/{expected} samples present -- shards incomplete", file=sys.stderr)
    sys.exit(3)
if errs > max_err:
    print(f"[agg] ABORT: {errs} errors > {max_err}", file=sys.stderr)
    sys.exit(3)
PY

export OUTPUT_DIR="$MERGED_DIR"
export SAMPLES_JSONL="$MERGED"
export SUMMARY_JSON="$OUTPUT_BASE/summary.json"
export NUM_SHARDS=1 SHARD_IDX=0
export RESUME=1 N_SAMPLES="${N_SAMPLES:-5}" SEED="${SEED:-42}"
export MAX_ERRORS SAVE_TEXT=1 ENABLE_THINKING=1
export MAX_TOKENS="${MAX_TOKENS:-32768}"
# Dead endpoint: with a complete merge the task list is empty and nothing is
# requested; <=MAX_ERRORS residual errors fail fast and stay errors (scored 0).
export VLLM_BASE_URL="http://127.0.0.1:9/v1"
export SERVED_MODEL_NAME="agg-noop"
export REQUEST_TIMEOUT=30
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline PYTHONUNBUFFERED=1

if [ "$MODE" = "research" ]; then
  RESEARCH_DATA="${RESEARCH_DATA:-$PROJECT_ROOT/data/frontiercs/research.parquet}"
  source "$PROJECT_ROOT/.venv/bin/activate"
  export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$PROJECT_ROOT/ALE-Bench/src:$PROJECT_ROOT/scripts:$PROJECT_ROOT/.cache/Frontier-CS-official:$PROJECT_ROOT/.cache/Frontier-CS-official/src${PYTHONPATH:+:$PYTHONPATH}"
  export FRONTIERCS_RESEARCH_PROMPT_STYLE="${FRONTIERCS_RESEARCH_PROMPT_STYLE:-concat}"
  python scripts/eval_qwen35_base_vllm_request.py \
    --source research \
    --research-data "$RESEARCH_DATA" \
    --num-shards 1 --shard-idx 0 \
    --output-dir "$OUTPUT_DIR" \
    --samples-jsonl "$SAMPLES_JSONL" \
    --summary-json "$SUMMARY_JSON" \
    --n-samples "$N_SAMPLES" \
    --base-url "$VLLM_BASE_URL" \
    --model "$SERVED_MODEL_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --presence-penalty 1.5 --repetition-penalty 1.0 \
    --concurrency 4 --timeout "$REQUEST_TIMEOUT" \
    --seed "$SEED" --max-errors "$MAX_ERRORS" --save-text \
    --enable-thinking --resume
else
  # fcsale: the standard wrapper (SOURCE=both, official prompt+scorer defaults).
  export SOURCE=both
  export FRONTIERCS_JUDGE_URL="http://127.0.0.1:9"
  export CONCURRENCY=4
  bash scripts/eval_base_model_qwen35_9b_vllm_request.sh
fi

echo "[agg] DONE -> $SUMMARY_JSON"
