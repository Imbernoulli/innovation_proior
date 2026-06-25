#!/bin/bash
DIR=/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-knapsack-negzero/verify
SOL=/tmp/cpv4b-dp-knapsack-negzero_sol
TMP=$DIR/_kz/in.txt
mismatch=0; total=0
for s in $(seq 1 600); do
  python3 "$DIR/gen.py" "$s" > "$TMP"
  out_sol=$("$SOL" < "$TMP")
  out_bru=$(python3 "$DIR/brute.py" < "$TMP")
  total=$((total+1))
  if [ "$out_sol" != "$out_bru" ]; then
    mismatch=$((mismatch+1))
    echo "MISMATCH seed=$s : sol=[$out_sol] brute=[$out_bru]"
    cat "$TMP"
    if [ $mismatch -ge 5 ]; then break; fi
  fi
done
echo "TOTAL=$total MISMATCHES=$mismatch"
