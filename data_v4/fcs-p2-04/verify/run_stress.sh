#!/bin/bash
set -e
N=${1:-600}
mismatch=0
for i in $(seq 1 $N); do
  python3 gen.py $i > in.txt
  out_sol=$(./sol < in.txt)
  out_bru=$(python3 brute.py < in.txt)
  if [ "$out_sol" != "$out_bru" ]; then
    echo "MISMATCH seed=$i sol=$out_sol brute=$out_bru"
    cat in.txt
    mismatch=$((mismatch+1))
    if [ $mismatch -ge 5 ]; then break; fi
  fi
done
echo "TOTAL_MISMATCHES=$mismatch over $N cases"
