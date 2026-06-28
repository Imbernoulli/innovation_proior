#!/bin/bash
set -e
fail=0
count=0
# Exhaustive cross-check on tiny + rand (small n) modes
for mode in tiny rand small smallpos greedytrap; do
  for seed in $(seq 1 120); do
    python3 gen.py $seed $mode > /tmp/c.in
    s=$(./sol < /tmp/c.in)
    b=$(python3 brute.py < /tmp/c.in)
    count=$((count+1))
    if [ "$s" != "$b" ]; then
      echo "MISMATCH sol/brute mode=$mode seed=$seed: sol=$s brute=$b"; cat /tmp/c.in; fail=$((fail+1))
    fi
    # exhaustive only on small n inputs
    nn=$(head -1 /tmp/c.in)
    if [ "$nn" -le 12 ]; then
      e=$(python3 exhaustive.py < /tmp/c.in)
      if [ "$e" != "$b" ]; then
        echo "MISMATCH brute/exh mode=$mode seed=$seed: brute=$b exh=$e"; cat /tmp/c.in; fail=$((fail+1))
      fi
    fi
  done
done
# Larger n: sol vs brute only (brute is the independent DP)
for mode in mid big extreme; do
  for seed in $(seq 1 50); do
    python3 gen.py $seed $mode > /tmp/c.in
    s=$(./sol < /tmp/c.in)
    b=$(python3 brute.py < /tmp/c.in)
    count=$((count+1))
    if [ "$s" != "$b" ]; then
      echo "MISMATCH sol/brute mode=$mode seed=$seed: sol=$s brute=$b"; fail=$((fail+1))
    fi
  done
done
# Edge modes
for mode in edge0 edge1; do
  for seed in $(seq 1 30); do
    python3 gen.py $seed $mode > /tmp/c.in
    s=$(./sol < /tmp/c.in)
    b=$(python3 brute.py < /tmp/c.in)
    count=$((count+1))
    if [ "$s" != "$b" ]; then
      echo "MISMATCH sol/brute mode=$mode seed=$seed: sol=$s brute=$b"; cat /tmp/c.in; fail=$((fail+1))
    fi
  done
done
echo "TOTAL CASES=$count  MISMATCHES=$fail"
