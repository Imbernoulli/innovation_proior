#!/usr/bin/env bash
# Parallel Codex review gate: ONE independent Codex session per method, all in parallel.
# (vs codex_gate_run.sh which is sequential in a single loop.)
# Usage: bash tools/codex_gate_parallel.sh <slug> [<slug> ...]
set -u
REPO="/Users/moonshot/paper2reasoning"
RUBRIC="/Users/moonshot/.claude/skills/paper-to-reasoning/SKILL.md"
LOGDIR="$REPO/tools/codex_logs"
cd "$REPO" || exit 1
mkdir -p "$LOGDIR"
COMPANION="$(ls ~/.claude/plugins/cache/*/codex/*/scripts/codex-companion.mjs 2>/dev/null | head -1)"
[ -z "${COMPANION:-}" ] && { echo "NO codex-companion.mjs"; exit 2; }

ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }

gate_one() {
  local slug="$1" d="methods/$1/results" log="$LOGDIR/$1.log"
  [ -f "$d/reasoning.md" ] || { echo "[$slug] SKIP no reasoning.md"; return; }
  local out rc
  out="$(node "$COMPANION" task \
    "Review AND FIX in place the paper-to-reasoning deliverables for $slug at $d/ (context.md, reasoning.md, answer.md). Rubric: $RUBRIC. Fix every issue directly in the files, preserving the in-frame style (English; reasoning.md continuous first-person, no markdown headers, discovery order, no meta-commentary; do NOT name or cite the target paper as a published artifact, but prior-art ancestors may be cited). Prioritize MATH/DERIVATION correctness (signs, factors, constants, every case of a case-analysis), then code faithfulness to the canonical implementation, then posterior/hindsight leaks, then scaffold purity and meta-commentary. Output a concise file:line changelog." \
    --write --model gpt-5.5 --effort xhigh 2>&1)"
  rc=$?
  printf '%s\n' "$out" > "$log"
  local reviewed outcome reason
  if printf '%s' "$out" | grep -qiE 'usage limit|rate limit|quota|429|too many requests|limit reached|will reset'; then
    reviewed=false; outcome=not_run; reason="codex usage/rate limit during parallel gate"
  elif [ $rc -eq 0 ] && printf '%s' "$out" | grep -qiE 'Turn completed|File changes completed|Changelog|file:[0-9]'; then
    reviewed=true; outcome=completed; reason=""
  else
    reviewed=false; outcome=failed; reason="codex exited rc=$rc without a confirmed changelog"
  fi
  python3 - "$slug" "$reviewed" "$outcome" "$reason" "$(ts)" <<'PY'
import json,sys
slug,reviewed,outcome,reason,now=sys.argv[1:6]
rec={"method":slug,"codex_reviewed":(reviewed=="true"),"outcome":outcome,
     "reviewed_at":now,"reviewer":"gpt-5.5","effort":"xhigh","evidence":"central-gate-parallel"}
if reason: rec["reason"]=reason
open(f"methods/{slug}/results/.codex_review.json","w").write(json.dumps(rec,indent=2)+"\n")
PY
  echo "[$slug] $outcome $(ts)"
}

MAXJOBS="${MAXJOBS:-8}"   # cap concurrent Codex sessions (one per task, throttled)
echo "=== parallel gate start $(ts) — MAXJOBS=$MAXJOBS — $# methods ==="
for slug in "$@"; do
  gate_one "$slug" &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
done
wait
echo "=== parallel gate done $(ts) ==="
