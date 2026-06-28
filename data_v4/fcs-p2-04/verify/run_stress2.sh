#!/bin/bash
mismatch=0; total=0
for i in $(seq 1001 1700); do
  python3 gen.py $i > in.txt
  os=$(./sol < in.txt); ob=$(python3 brute.py < in.txt)
  total=$((total+1))
  if [ "$os" != "$ob" ]; then echo "MISMATCH seed=$i sol=$os brute=$ob"; cat in.txt; mismatch=$((mismatch+1)); [ $mismatch -ge 5 ] && break; fi
done
echo "BATCH2 TOTAL_MISMATCHES=$mismatch over $total cases"
