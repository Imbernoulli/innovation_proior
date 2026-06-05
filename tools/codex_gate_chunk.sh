#!/bin/bash
# One parallel codex-gate worker over a chunk of methods.
# Usage: codex_gate_chunk.sh <tag> <chunkfile>   (chunkfile = one slug per line)
# - Unique CODEX_COMPANION_SESSION_ID per worker so concurrent workers don't
#   collide on the companion's "one active task per session" guard.
# - Sequential within the worker; codex task is blocking (timeout 1500).
# - Sentinel results/.codex_done => skip (resumable). Rate-limit => backoff+retry.
set -u
TAG="$1"; CHUNK="$2"
export CODEX_COMPANION_SESSION_ID="codexgate-${TAG}-$$"
BASE=/Users/moonshot/paper2reasoning/methods
COMP=/Users/moonshot/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs
SKILL=/Users/moonshot/.claude/skills/paper-to-reasoning/SKILL.md
LOGDIR=/tmp/codexgate; mkdir -p "$LOGDIR"
REPORT="$LOGDIR/report-${TAG}.txt"; : > "$REPORT"

prompt_for () {
  local m="$1"
  echo "Review AND FIX in place the paper-to-reasoning deliverables for ${m} at ${BASE}/${m}/results/ (context.md, reasoning.md, answer.md). Rubric: ${SKILL}. Fix every issue directly in the files, preserving in-frame style (English; reasoning.md continuous first-person, ZERO markdown headers outside code fences, discovery order, no meta-commentary; NEVER reference the source paper as a published artifact — the narrator IS the inventor). Prioritize MATH/DERIVATION correctness (signs, factors, constants, every case of a case-analysis), then code faithfulness to the canonical implementation, then posterior/hindsight leaks, then scaffold purity (the context.md code stubs must correspond piece-for-piece to the final reasoning/answer code, NOT mirror the external repo), then meta-commentary. Output a concise file:line changelog."
}
is_rate_limited () { grep -qiE "usage limit|rate limit|try again at|429|too many requests|quota" "$1"; }

while IFS= read -r m; do
  [ -z "$m" ] && continue
  if [ -f "$BASE/$m/results/.codex_done" ]; then
    echo "[$(date '+%T')] skip $m (done)" | tee -a "$REPORT"; continue
  fi
  attempt=0
  while true; do
    attempt=$((attempt+1))
    echo "[$(date '+%T')] codex $m (attempt $attempt)" | tee -a "$REPORT"
    timeout 1700 node "$COMP" task "$(prompt_for "$m")" --write --model gpt-5.5 --effort xhigh > "$LOGDIR/${m}.log" 2>&1
    if is_rate_limited "$LOGDIR/${m}.log"; then
      echo "[$(date '+%T')]   rate-limited; sleep 600 then retry $m" | tee -a "$REPORT"
      sleep 600; continue
    fi
    touch "$BASE/$m/results/.codex_done"
    echo "[$(date '+%T')]   DONE $m" | tee -a "$REPORT"
    break
  done
done < "$CHUNK"
echo "[$(date '+%F %T')] chunk ${TAG} COMPLETE" | tee -a "$REPORT"
