#!/usr/bin/env bash
# jiaolab MLS-Bench CPU baseline: N full 22-task rollouts + the fixed-denominator
# aggregate. This is the MLS half of scripts/gpublaze/run_4b_base_mls_research_baseline.sh
# (same tools gate, same rollout loop, same aggregation) with the FrontierCS-research
# phase left out on purpose -- research on this box goes through
# scripts/jiaolab/eval_client_local.sh (SOURCE=research), which is a different
# stack with its own readiness story (docs/EVAL_ON_JIAOLAB_zh.md §2.3).
#
# A single rollout is NOT the published protocol: MLS-Bench CPU is noisy and the
# gpublaze numbers are the mean over N>=3 rollouts, with the denominator PINNED
# at 22 (DEFAULT_CPU_TASKS) so unscored/failed tasks count as 0. Comparing a
# 1-rollout jiaolab number with a 3-rollout gpublaze number is not a comparison.
#
# Attach to an already-running engine (no GPUs claimed by us):
#   EXTERNAL_VLLM_URL=http://127.0.0.1:8006/v1 SERVED=qwen35-4b-base \
#     setsid nohup bash scripts/jiaolab/run_mls_cpu_baseline.sh >> logs/mls_base.log 2>&1 &
# Or let it own a card:
#   GPUS=2 MODEL_PATH=<dir> SERVED=<tag> bash scripts/jiaolab/run_mls_cpu_baseline.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_jiaolab.sh"
cd "$FS_ROOT"

SERVED="${SERVED:-${TAG:?set SERVED (the served-model-name / output tag)}}"
N_ROLLOUTS="${N_ROLLOUTS:-3}"
MLS_OUT_ROOT="${MLS_OUT_ROOT:-$FS_ROOT/outputs/mls_cpu_base_${SERVED}}"
DRIVER_PY="${MLSBENCH_PY:-/home/bohan/miniconda3/envs/mlsbench-driver/bin/python}"
[ -x "$DRIVER_PY" ] || DRIVER_PY="$(command -v python3)"

log() { echo "[mls-chain $(date '+%F %T')] $*"; }

# ---- rollouts (PHASES=aggregate re-aggregates existing runs without evaluating) --
if [[ "${PHASES:-rollouts,aggregate}" == *rollouts* ]]; then
  # eval_mlsbench_local.sh runs its own preflight and its own tool-call parse gate
  # (qwen3_xml) on every rollout, so a backend that silently stops parsing tool
  # calls mid-baseline stops the chain instead of writing 0/22 rows.
  for i in $(seq 1 "$N_ROLLOUTS"); do
    if [ -f "$MLS_OUT_ROOT/run$i/summary.json" ] && [ "${RESUME:-1}" = "1" ]; then
      log "rollout $i/$N_ROLLOUTS already has a summary.json; skipping (RESUME=0 to force)"
      continue
    fi
    log "MLS rollout $i/$N_ROLLOUTS -> $MLS_OUT_ROOT/run$i"
    TAG="$SERVED" OUTPUT_BASE="$MLS_OUT_ROOT/run$i" \
    CONCURRENCY="${CONCURRENCY:-20}" TASK_TIMEOUT="${TASK_TIMEOUT:-5400}" \
      bash "$SCRIPT_DIR/eval_mlsbench_local.sh" \
      >> "$FS_ROOT/logs/mls_base_${SERVED}_run$i.log" 2>&1
    log "MLS rollout $i done rc=$? (summary: $MLS_OUT_ROOT/run$i/summary.json)"
  done
fi

# ---- aggregate ----------------------------------------------------------------
"$DRIVER_PY" - "$MLS_OUT_ROOT" "$N_ROLLOUTS" <<'PY'
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
    "node": "jiaolab",
    "denominator": DENOM,
    "rollouts": runs,
    "per_task_mean_over_runs": {k: sum(v)/len(v) for k, v in sorted(per_task.items())},
    "overall_mean_of_run_means_fixed_denom": (
        sum(sum(v)/len(v) for v in per_task.values()) / DENOM if per_task else None),
    "note": "unscored/failed tasks count as 0 in the fixed-denominator mean; "
            "jiaolab (Xeon 8358) numbers are NOT same-table comparable with gpublaze (EPYC 9654)",
}
out = root / "aggregate.json"
out.write_text(json.dumps(agg, indent=1))
print(f"[mls-chain] aggregate -> {out}")
print(json.dumps(agg, indent=1))
PY
log "CHAIN COMPLETE"
