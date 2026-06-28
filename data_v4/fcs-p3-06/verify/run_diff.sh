#!/bin/bash
set -e
modes=(tiny small mid edge mixed)
mismatch=0
total=0
for seed in $(seq 1 110); do
  for mode in "${modes[@]}"; do
    python3 gen.py $seed $mode > inp.txt
    ./sol < inp.txt > out_sol.txt
    python3 brute.py < inp.txt > out_brute.txt
    total=$((total+1))
    if ! diff -q out_sol.txt out_brute.txt > /dev/null; then
      echo "MISMATCH seed=$seed mode=$mode"
      echo "--- input ---"; cat inp.txt
      echo "--- sol ---"; cat out_sol.txt
      echo "--- brute ---"; cat out_brute.txt
      mismatch=$((mismatch+1))
      if [ $mismatch -ge 3 ]; then exit 1; fi
    fi
  done
done
echo "TOTAL=$total MISMATCH=$mismatch"
