#!/usr/bin/env bash
set -u
SCRATCH=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad
mkdir -p "$SCRATCH"
V=/srv/home/bohanlyu/innovation_proior/data_v4/cpv4-graph-bfs-count/verify
IN="$SCRATCH/cpv4_in_$$.txt"
N=${1:-400}
START=${2:-1}
mismatch=0; total=0
for seed in $(seq "$START" $((START + N - 1))); do
  python3 "$V/gen.py" "$seed" > "$IN"
  out_sol=$(/tmp/cpv4-graph-bfs-count_sol < "$IN")
  out_brute=$(python3 "$V/brute.py" < "$IN")
  total=$((total+1))
  if [ "$out_sol" != "$out_brute" ]; then
    mismatch=$((mismatch+1))
    if [ $mismatch -le 6 ]; then
      echo "MISMATCH seed=$seed sol=[$out_sol] brute=[$out_brute]"
      echo "--- input ---"; cat "$IN"; echo "-------------"
    fi
  fi
done
echo "TOTAL=$total MISMATCHES=$mismatch"
rm -f "$IN"
