#!/bin/bash
# Launch ONE Qwen3.8-27B vLLM service. TP = number of GPUs passed (2 or 4).
# MEASURED 2026-08-15 on H100 NVLink (4 cards, 64k ctx, hard-CP rollout, 60s windows):
#   2 x TP=2 @96 total in-flight  : 3,567 tok/s aggregate, KV 0.68-0.79
#   1 x TP=4 @96                  : 3,124 tok/s (-12%: all-reduce sync idles a card)
#   1 x TP=4 @144                 : 4,298 tok/s (+20%), KV 0.51, 0 preemption
#   1 x TP=4 @192                 : 4,543 tok/s at fresh KV, but MATURED KV -> 1.00, thrash, 1,575 tok/s
#   OPERATING POINT = TP=4 @144 (0 preempt after KV matures; measure AFTER maturation, not at fill)
# WHY TP=4 wins: one weight copy instead of two frees ~26GB -> KV pool 4,378 blocks vs 2x1,696=3,392
# (+29%), so it sustains ~2x the concurrency of a single TP=2 replica; extra batch amortizes the
# all-reduce cost. Same-concurrency TP=4 is slower — the win ONLY appears at higher in-flight.
# 64k ctx / 57k max_tokens: a 131k/110k experiment collapsed KV to ~14 per replica; don't.
GPUS="$1"; PORT="${2:-30002}"
TP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
SEQS=$(( TP == 4 ? 448 : 224 ))
SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad
mkdir -p "$SC"
CUDA_VISIBLE_DEVICES="$GPUS" VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 \
setsid nohup /srv/home/bohanlyu/sesl/.venv/bin/vllm serve Qwen/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" --dtype bfloat16 \
  --max-model-len 65536 --max-num-seqs "$SEQS" --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --trust-remote-code --enable-prefix-caching \
  >> "$SC/vllm_$PORT.log" 2>&1 &
echo "launched Qwen3.8-27B TP=$TP service on GPUs $GPUS port $PORT (64k ctx, max-num-seqs $SEQS)"
