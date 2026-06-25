#!/usr/bin/env bash
set -u
DIR="/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-knapsack-precision/verify"
SD="/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad"
mkdir -p "$SD"
SOL="/tmp/cpv4b-dp-knapsack-precision_sol"
N="${1:-400}"
fail=0
for i in $(seq 1 "$N"); do
  python3 "$DIR/gen.py" "$i" > "$SD/in_$i.txt"
  out_sol=$("$SOL" < "$SD/in_$i.txt")
  out_brute=$(python3 "$DIR/brute.py" < "$SD/in_$i.txt")
  if [ "$out_sol" != "$out_brute" ]; then
    echo "MISMATCH seed=$i"
    cat "$SD/in_$i.txt"
    echo "SOL=$out_sol BRUTE=$out_brute"
    fail=$((fail+1))
    if [ "$fail" -ge 5 ]; then break; fi
  fi
  rm -f "$SD/in_$i.txt"
done
echo "TOTAL_MISMATCHES=$fail over $N cases"
