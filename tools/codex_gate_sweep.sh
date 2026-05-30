#!/bin/bash
# Resumable independent-Codex gate over the 92 new methods.
# - Sequential (quota-bound; concurrency wouldn't increase throughput once rate-limited).
# - On usage/rate limit: back off 10 min and retry the SAME method (self-discovers reset).
# - On success: touch results/.codex_done so re-runs skip it. Re-launch this script to resume.
set -u
BASE=/Users/moonshot/paper2reasoning/methods
COMP=/Users/moonshot/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs
SKILL=/Users/moonshot/.claude/skills/paper-to-reasoning/SKILL.md
LOGDIR=/tmp/codexgate
REPORT=/Users/moonshot/paper2reasoning/tools/codex_gate_report.txt
mkdir -p "$LOGDIR"
: > "$REPORT"

METHODS="adafactor alibi alphazero barlow-twins beit bigbird biggan c51 chebnet classifier-free-guidance consistency-models cpc cql decision-transformer deep-sets detr dino double-dqn dpm-solver dreamer dueling-dqn edm efficientnet electra elmo flow-matching fno focal-loss fpn gae gail gin glow googlenet gqa graphormer group-norm gumbel-softmax her hippo hyena impala iwae kfac lamb lars linear-attention longformer maml mask-rcnn mixup mobilenet moe mpnn muon muzero nerf neural-ode ntk performer pix2pix pixelcnn prioritized-replay progressive-gan radam rectified-flow reformer retnet rmsnorm s4 sam senet shampoo simsiam sophia spatial-transformer spectral-norm speculative-decoding ssd stylegan2 swav swin switch-transformer transformer-xl vdm vicreg vqgan vqvae wavenet weight-norm xlnet yolo"

run_one () {
  local m="$1"
  local prompt="Review AND FIX in place the paper-to-reasoning deliverables for ${m} at ${BASE}/${m}/results/ (context.md, reasoning.md, answer.md). Rubric: ${SKILL}. Fix every issue directly in the files, preserving in-frame style (English; reasoning.md continuous first-person, ZERO markdown headers outside code fences, discovery order, no meta-commentary; NEVER reference the source paper as a published artifact — the narrator IS the inventor discovering this method). Prioritize MATH/DERIVATION correctness (signs, factors, constants, every case of a case-analysis), then code faithfulness to the canonical implementation, then posterior/hindsight leaks, then scaffold purity (the context.md code stubs must correspond piece-for-piece to the final reasoning/answer code, NOT mirror the external repo), and meta-commentary. Output a concise file:line changelog."
  timeout 1500 node "$COMP" task "$prompt" --write --model gpt-5.5 --effort xhigh > "$LOGDIR/${m}.log" 2>&1
}

is_rate_limited () {
  grep -qiE "usage limit|rate limit|try again at|429|too many requests|quota" "$1"
}

total=0; done_n=0
for m in $METHODS; do total=$((total+1)); done
echo "[$(date '+%F %T')] starting codex gate sweep over $total methods" | tee -a "$REPORT"

for m in $METHODS; do
  if [ -f "$BASE/$m/results/.codex_done" ]; then
    done_n=$((done_n+1)); echo "[$(date '+%T')] skip $m (already gated)" | tee -a "$REPORT"; continue
  fi
  attempt=0
  while true; do
    attempt=$((attempt+1))
    echo "[$(date '+%T')] codex $m (attempt $attempt)" | tee -a "$REPORT"
    run_one "$m"
    if is_rate_limited "$LOGDIR/${m}.log"; then
      echo "[$(date '+%T')]   rate-limited; sleeping 600s then retrying $m" | tee -a "$REPORT"
      sleep 600
      continue
    fi
    touch "$BASE/$m/results/.codex_done"
    done_n=$((done_n+1))
    cl=$(grep -iE "changelog|changed|fixed|edited|no changes|clean" "$LOGDIR/${m}.log" | head -1 | cut -c1-100)
    echo "[$(date '+%T')]   DONE $m ($done_n/$total) :: ${cl}" | tee -a "$REPORT"
    break
  done
done

echo "[$(date '+%F %T')] codex gate sweep COMPLETE: $done_n/$total gated" | tee -a "$REPORT"
