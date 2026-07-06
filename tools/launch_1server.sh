#!/bin/bash
# Launch ONE independent Qwen3.6-27B TP=2 vLLM service on a GPU pair + port, with prefix caching
# (so a query pinned to this service reuses its prompt's KV across all its samples). P2P/NVLink ON.
GPUS="$1"; PORT="$2"
SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad
CUDA_VISIBLE_DEVICES="$GPUS" VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 \
setsid nohup /srv/home/bohanlyu/sesl/.venv/bin/vllm serve Qwen/Qwen3.6-27B \
  --served-model-name Qwen3.6-27B --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 65536 --max-num-seqs 224 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --trust-remote-code --enable-prefix-caching \
  >> "$SC/vllm_$PORT.log" 2>&1 &
echo "launched TP=2 service on GPUs $GPUS port $PORT (prefix-caching on)"
