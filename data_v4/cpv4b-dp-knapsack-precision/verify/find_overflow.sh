#!/usr/bin/env bash
set -u
DIR="/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-knapsack-precision/verify"
SD="/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad"
mkdir -p "$SD"
g++ -O2 -std=c++17 -o "$SD/buggy" "$DIR/sol_buggy.cpp" || exit 1
SOL="/tmp/cpv4b-dp-knapsack-precision_sol"
diffs=0
for i in $(seq 1 2000); do
  python3 "$DIR/gen.py" "$i" > "$SD/oin.txt"
  a=$("$SOL" < "$SD/oin.txt")
  b=$("$SD/buggy" < "$SD/oin.txt")
  if [ "$a" != "$b" ]; then
    echo "OVERFLOW-DIVERGENCE at seed=$i : correct=$a  buggy=$b"
    cat "$SD/oin.txt"
    diffs=$((diffs+1))
    if [ "$diffs" -ge 3 ]; then break; fi
  fi
done
echo "DIVERGENCES_FOUND=$diffs"
rm -f "$SD/oin.txt"
