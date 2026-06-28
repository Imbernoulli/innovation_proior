#!/bin/bash
set -e
SOL=./sol
mismatch=0
count=0
for seed in $(seq 1 700); do
  python3 gen.py $seed > in.txt
  out_sol=$(./sol < in.txt)
  out_brute=$(python3 brute.py < in.txt)
  count=$((count+1))
  if [ "$out_sol" != "$out_brute" ]; then
    echo "MISMATCH seed=$seed"
    echo "--- input ---"; cat in.txt
    echo "sol=$out_sol brute=$out_brute"
    mismatch=$((mismatch+1))
    if [ $mismatch -ge 5 ]; then break; fi
  fi
done
echo "RANDOM: ran $count cases, mismatches=$mismatch"
