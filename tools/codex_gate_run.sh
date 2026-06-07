#!/usr/bin/env bash
# Central sequential Codex review gate for paper-to-reasoning deliverables.
# Runs Codex (gpt-5.5, xhigh) over each method's results/ ONE AT A TIME — sequential
# so we respect Codex usage quota and never cascade into rate-limits. On a genuine pass it
# writes methods/<slug>/results/.codex_review.json (codex_reviewed:true); on a limit/error it
# writes codex_reviewed:false / not_run|failed with the reason, so the gap stays visible.
#
# Usage: bash tools/codex_gate_run.sh <slug> [<slug> ...]
set -u
REPO="/Users/moonshot/paper2reasoning"
RUBRIC="/Users/moonshot/.claude/skills/paper-to-reasoning/SKILL.md"
LOG="$REPO/tools/codex_gate_run.log"
cd "$REPO" || exit 1

COMPANION="$(ls ~/.claude/plugins/cache/*/codex/*/scripts/codex-companion.mjs 2>/dev/null | head -1)"
if [ -z "${COMPANION:-}" ]; then echo "NO codex-companion.mjs found" | tee -a "$LOG"; exit 2; fi

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
marker() { # slug reviewed outcome reason
  python3 - "$1" "$2" "$3" "$4" "$(ts)" <<'PY'
import json,sys
slug,reviewed,outcome,reason,now=sys.argv[1:6]
rec={"method":slug,"codex_reviewed":(reviewed=="true"),"outcome":outcome,
     "reviewed_at":now,"reviewer":"gpt-5.5","effort":"xhigh","evidence":"central-gate"}
if reason: rec["reason"]=reason
open(f"methods/{slug}/results/.codex_review.json","w").write(json.dumps(rec,indent=2)+"\n")
PY
}

echo "=== gate start $(ts) — slugs: $* ===" | tee -a "$LOG"
for slug in "$@"; do
  d="methods/$slug/results"
  if [ ! -f "$d/reasoning.md" ]; then echo "[$slug] SKIP (no reasoning.md)" | tee -a "$LOG"; continue; fi
  echo "--- [$slug] codex start $(ts) ---" | tee -a "$LOG"
  out="$(node "$COMPANION" task \
    "Review AND FIX in place the paper-to-reasoning deliverables for $slug at $d/ (context.md, reasoning.md, answer.md). Rubric: $RUBRIC. Fix every issue directly in the files, preserving the in-frame style (English; reasoning.md continuous first-person, no markdown headers, discovery order, no meta-commentary; do NOT name or cite the target paper as a published artifact, but prior-art ancestors may be cited). Prioritize MATH/DERIVATION correctness (signs, factors, constants, every case of a case-analysis), then code faithfulness to the canonical implementation, then posterior/hindsight leaks, then scaffold purity and meta-commentary. Output a concise file:line changelog." \
    --write --model gpt-5.5 --effort xhigh 2>&1)"
  rc=$?
  echo "$out" | tail -40 >> "$LOG"
  if echo "$out" | grep -qiE 'usage limit|rate limit|quota|429|too many requests|limit reached|will reset'; then
    echo "[$slug] LIMITED $(ts)" | tee -a "$LOG"; marker "$slug" false not_run "codex usage/rate limit during central gate"
  elif [ $rc -eq 0 ] && echo "$out" | grep -qiE 'Turn completed|File changes completed|Changelog|file:[0-9]'; then
    echo "[$slug] REVIEWED $(ts)" | tee -a "$LOG"; marker "$slug" true completed ""
  else
    echo "[$slug] FAILED rc=$rc $(ts)" | tee -a "$LOG"; marker "$slug" false failed "codex exited rc=$rc without a confirmed changelog"
  fi
done
echo "=== gate done $(ts) ===" | tee -a "$LOG"
