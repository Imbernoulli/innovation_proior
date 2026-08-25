#!/usr/bin/env bash
# 4B-base baseline chain on gpublaze (2026-08-23, coordinator order):
#   phase 0  wait until the shared vLLM (EXTERNAL_VLLM_URL, default :8006) accepts
#            a `tools` request -- the MLS agent needs the server started with
#            --enable-auto-tool-choice --tool-call-parser hermes (raw-think口径
#            unaffected: the tool parser is orthogonal to any reasoning parser)
#   phase 1  MLS-Bench CPU baseline: N_ROLLOUTS (>=3) full 22-task runs
#   phase 2  FrontierCS research runnable-subset (43 minus 20 cant_be_late*;
#            symbolic_regression dropped too when pysr/Julia is not importable),
#            REPS=3 on lottery families, denominator recorded in a NOTE file
#   phase 3  aggregate MLS runs (fixed denominator 22, mean-of-runs per task)
#
# Run detached:  setsid nohup bash scripts/gpublaze/run_4b_base_mls_research_baseline.sh \
#                  >> logs/base4b_mls_research_chain.log 2>&1 &
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_gpublaze.sh"
cd "$FS_ROOT"

EXTERNAL_VLLM_URL="${EXTERNAL_VLLM_URL:-http://127.0.0.1:8006/v1}"
SERVED="${SERVED:-qwen35-4b-base}"
N_ROLLOUTS="${N_ROLLOUTS:-3}"
WAIT_TOOLS_SEC="${WAIT_TOOLS_SEC:-86400}"
MLS_OUT_ROOT="$FS_ROOT/outputs/mls_cpu_base_${SERVED}"
RES_OUT="$FS_ROOT/outputs/cc_eval_${SERVED}_research_runnable_thinking_32k_vllm"

log() { echo "[chain $(date '+%F %T')] $*"; }

# The gate must prove the server PARSES tool calls, not merely accepts `tools`:
# the 2026-08-23 hermes run accepted every request yet parsed nothing (Qwen3.5
# emits the XML format -> needs --tool-call-parser qwen3_xml), which produced a
# silent 0/22 in ~3 min. Deterministic probe: temperature 0, instruct a test()
# call, require a non-empty parsed tool_calls array in the response.
tools_ok() {
  curl -sS --max-time 240 "${EXTERNAL_VLLM_URL%/}/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"temperature\":0,\"max_tokens\":4000,\"messages\":[{\"role\":\"system\",\"content\":\"Always respond with a tool call.\"},{\"role\":\"user\",\"content\":\"Run a first experiment now by calling the test tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"test\",\"description\":\"Run a new experiment\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}" \
    2>/dev/null | python3 -c "import json,sys
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls') or []
    sys.exit(0 if tc else 1)
except Exception:
    sys.exit(1)"
}

PHASES="${PHASES:-mls,research}"

# ---- phase 0: wait until tool calls actually PARSE ----------------------------
if [[ "$PHASES" == *mls* ]]; then
  start=$SECONDS
  until tools_ok; do
    if (( SECONDS - start >= WAIT_TOOLS_SEC )); then
      log "GIVING UP: $EXTERNAL_VLLM_URL never returned parsed tool_calls after ${WAIT_TOOLS_SEC}s."
      log "restart it with: --enable-auto-tool-choice --tool-call-parser qwen3_xml"
      exit 1
    fi
    log "waiting: $EXTERNAL_VLLM_URL returns no parsed tool_calls (server needs --enable-auto-tool-choice --tool-call-parser qwen3_xml); retry in 60s"
    sleep 60
  done
  log "backend parses tool calls; starting MLS baseline (${N_ROLLOUTS} rollouts x 22 tasks)"
fi

# ---- phase 1: MLS CPU baseline ------------------------------------------------
[[ "$PHASES" == *mls* ]] && \
for i in $(seq 1 "$N_ROLLOUTS"); do
  log "MLS rollout $i/$N_ROLLOUTS -> $MLS_OUT_ROOT/run$i"
  EXTERNAL_VLLM_URL="$EXTERNAL_VLLM_URL" TAG="$SERVED" \
  OUTPUT_BASE="$MLS_OUT_ROOT/run$i" \
  CONCURRENCY="${CONCURRENCY:-20}" TASK_TIMEOUT="${TASK_TIMEOUT:-5400}" \
    bash "$SCRIPT_DIR/eval_mlsbench_local.sh" \
    >> "$FS_ROOT/logs/mls_base_${SERVED}_run$i.log" 2>&1
  rc=$?
  log "MLS rollout $i done rc=$rc (summary: $MLS_OUT_ROOT/run$i/summary.json)"
done

# ---- phase 3 (early, cheap): aggregate MLS ------------------------------------
[[ "$PHASES" == *mls* ]] && \
"$FS_CLIENT_VENV/bin/python" - "$MLS_OUT_ROOT" "$N_ROLLOUTS" <<'PY'
import json, sys
from pathlib import Path
root, n = Path(sys.argv[1]), int(sys.argv[2])
DENOM = 22  # fixed denominator: DEFAULT_CPU_TASKS
per_task, runs = {}, []
for i in range(1, n + 1):
    p = root / f"run{i}" / "summary.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())
    runs.append({"run": i, "mean_score": s.get("mean_score"), "n_scored": s.get("n_scored")})
    for t in s.get("tasks", []):
        per_task.setdefault(t["task"], []).append(t.get("score") if t.get("status") == "scored" else 0.0)
agg = {
    "denominator": DENOM,
    "rollouts": runs,
    "per_task_mean_over_runs": {k: sum(v)/len(v) for k, v in sorted(per_task.items())},
    "overall_mean_of_run_means_fixed_denom": (
        sum(sum(v)/len(v) for v in per_task.values()) / DENOM if per_task else None),
    "note": "unscored/failed tasks count as 0 in the fixed-denominator mean",
}
out = root / "aggregate.json"
out.write_text(json.dumps(agg, indent=1))
print(f"[chain] MLS aggregate -> {out}")
PY

# ---- phase 2: research runnable subset ----------------------------------------
if [[ "$PHASES" != *research* ]]; then log "CHAIN COMPLETE (phases=$PHASES)"; exit 0; fi
RES_DATA="$FS_ROOT/data/frontiercs/research_cpu_runnable.parquet"
DENOM_NOTE="runnable subset = 43 CPU problems minus 20 cant_be_late/cant_be_late_multi (missing real_traces.tar.gz)"
if ! JULIA_DEPOT_PATH="$FS_ROOT/.cache/julia_depot" PYTHON_JULIAPKG_PROJECT="$FS_ROOT/.cache/julia_env" \
     "$FS_CLIENT_VENV/bin/python" -c "import pysr" >/dev/null 2>&1; then
  log "pysr/Julia not importable -> dropping symbolic_regression (5) from the subset"
  "$FS_CLIENT_VENV/bin/python" - <<'PY'
import pandas as pd
df = pd.read_parquet("data/frontiercs/research_cpu_runnable.parquet")
fam = df["reward_model"].apply(lambda r: r["ground_truth"]).str.split("/").str[0]
df[~fam.isin(["symbolic_regression"])].reset_index(drop=True).to_parquet(
    "data/frontiercs/research_cpu_runnable_nosr.parquet", index=False)
PY
  RES_DATA="$FS_ROOT/data/frontiercs/research_cpu_runnable_nosr.parquet"
  DENOM_NOTE="$DENOM_NOTE; minus 5 symbolic_regression (pysr/Julia unavailable)"
fi
NROWS=$("$FS_CLIENT_VENV/bin/python" -c "import pandas as pd; print(len(pd.read_parquet('$RES_DATA')))")
log "research subset: $NROWS problems ($DENOM_NOTE)"
mkdir -p "$RES_OUT/shard_0"
printf '{"denominator": %s, "note": "%s"}\n' "$NROWS" "$DENOM_NOTE" > "$RES_OUT/DENOMINATOR_NOTE.json"

export JULIA_DEPOT_PATH="$FS_ROOT/.cache/julia_depot" PYTHON_JULIAPKG_PROJECT="$FS_ROOT/.cache/julia_env"
TAG="${SERVED}_research_runnable" MODEL_TAG="$SERVED" SOURCE=research \
VLLM_BASE_URL="$EXTERNAL_VLLM_URL" RESEARCH_DATA="$RES_DATA" \
NUM_SHARDS=1 SHARD_IDX=0 RESUME=1 \
FRONTIERCS_RESEARCH_SCORE_REPS="${FRONTIERCS_RESEARCH_SCORE_REPS:-3}" \
FRONTIERCS_RESEARCH_REPS_ONLY="${FRONTIERCS_RESEARCH_REPS_ONLY:-vdb_pareto,symbolic_regression}" \
CONCURRENCY="${RESEARCH_CONCURRENCY:-32}" \
OUTPUT_BASE="$RES_OUT" \
  bash "$SCRIPT_DIR/eval_client_local.sh" \
  >> "$FS_ROOT/logs/research_base_${SERVED}.log" 2>&1
rc=$?
log "research subset done rc=$rc -> $RES_OUT/shard_0/summary_shard.json (denominator note: $RES_OUT/DENOMINATOR_NOTE.json)"
log "CHAIN COMPLETE"
