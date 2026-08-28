#!/usr/bin/env bash
# One-shot snapshot of every taste-eval line on both nodes.
cd /srv/home/bohanlyu/innovation_proior
echo "===== $(date '+%Y-%m-%d %H:%M') ====="
echo "--- gpublaze 4B re-run (corrected sampling) ---"
for f in logs_taste/redo_*.log; do [ -f "$f" ] || continue; printf "  %-16s " "$(basename $f .log)"; tail -n 1 "$f" | cut -c1-88; done
wc -l outputs_taste/run4b_redo/*.jsonl 2>/dev/null | grep -v total | sed 's|outputs_taste/run4b_redo/cc_eval_|    |'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' '|' ; echo
echo "--- jiaolab 9B family ---"
ssh -o ServerAliveInterval=20 -o ConnectTimeout=20 jiaolab 'cd /home/bohan/innovation_proior
for f in logs_taste/arm9b_*.log; do printf "  %-26s " "$(basename $f .log | sed s/arm9b_//)"; tail -n 1 "$f" | cut -c1-84; done
wc -l outputs_taste/run9b/*.jsonl 2>/dev/null | grep -v total | sed "s|outputs_taste/run9b/cc_eval_|    |"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" "|"; echo
df -h / | tail -1' 2>&1 | tail -30
echo "--- shipping ---"
tail -n 2 outputs_taste/ship9b_rest.log 2>/dev/null
