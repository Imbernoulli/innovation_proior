#!/usr/bin/env bash
set -u
DIR=/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-bitmask-precision/verify
SOL=/tmp/cpv4b-dp-bitmask-precision_sol
N=${1:-400}
pass=0; fail=0; first_fail=""
for s in $(seq 1 "$N"); do
  python3 "$DIR/gen.py" "$s" > /tmp/cpv4b_in.txt
  out_sol=$("$SOL" < /tmp/cpv4b_in.txt)
  out_bru=$(python3 "$DIR/brute.py" < /tmp/cpv4b_in.txt)
  if [ "$out_sol" == "$out_bru" ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    if [ -z "$first_fail" ]; then
      first_fail=$s
      cp /tmp/cpv4b_in.txt /tmp/cpv4b_fail.txt
      echo "MISMATCH seed=$s sol=$out_sol brute=$out_bru"
    fi
  fi
done
echo "PASS=$pass FAIL=$fail FIRST_FAIL=$first_fail"
