#!/usr/bin/env bash
DIR=/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-1d-greedytrap/verify
SOL=/tmp/cpv4b-dp-1d-greedytrap_sol
mismatch=0; total=0
N=${1:-400}
for seed in $(seq 1 $N); do
  python3 "$DIR/gen_big.py" $seed > /tmp/cpv4b_in_big.txt
  out_sol=$("$SOL" < /tmp/cpv4b_in_big.txt)
  out_bru=$(python3 "$DIR/brute.py" < /tmp/cpv4b_in_big.txt)
  total=$((total+1))
  if [ "$out_sol" != "$out_bru" ]; then
    mismatch=$((mismatch+1))
    echo "MISMATCH seed=$seed:"; cat /tmp/cpv4b_in_big.txt; echo "sol=$out_sol bru=$out_bru"
    if [ $mismatch -ge 5 ]; then break; fi
  fi
done
echo "TOTAL=$total MISMATCHES=$mismatch"
