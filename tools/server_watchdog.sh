#!/bin/bash
# Watchdog: keep the Qwen3.6-27B vLLM server (port 30000) alive. If it's down for two
# consecutive checks, kill any hung replica and relaunch. Only touches OUR process
# (`vllm serve Qwen/Qwen3.6-27B`) — never other users' sglang servers or GPUs.
SC=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad
LOG="$SC/vllm_server.log"
up() { curl -sf http://127.0.0.1:30000/v1/models >/dev/null 2>&1; }
launch() {
  pkill -9 -f 'vllm serve Qwen/Qwen3.6-27B' 2>/dev/null; sleep 8
  mv "$LOG" "$SC/vllm_server.$(date +%s 2>/dev/null || echo old).log" 2>/dev/null
  CUDA_VISIBLE_DEVICES=1,2,3,5,7 VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 NCCL_P2P_DISABLE=1 \
  setsid nohup /srv/home/bohanlyu/sesl/.venv/bin/vllm serve Qwen/Qwen3.6-27B \
    --served-model-name Qwen3.6-27B --host 0.0.0.0 --port 30000 \
    --data-parallel-size 5 --tensor-parallel-size 1 --dtype bfloat16 \
    --max-model-len 65536 --max-num-seqs 224 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --trust-remote-code >> "$LOG" 2>&1 &
  sleep 210  # allow checkpoint load + engine init before re-checking
}
while true; do
  if ! up; then
    sleep 10
    if ! up; then echo "[watchdog $(date -u 2>/dev/null)] server DOWN -> relaunch" >> "$SC/watchdog.log"; launch; fi
  fi
  sleep 30
done
