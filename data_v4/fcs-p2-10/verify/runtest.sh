#!/bin/bash
set -e
N_RAND=${1:-600}
mism=0
total=0
# Edge cases
edges=(
"0"
"1
0"
"1
5"
"2
0 0"
"2
3 3"
"2
1 5"
"3
1 5 8"
"4
1 5 8 9"
"5
2 5 7 8 10"
"5
0 0 0 0 0"
"6
1 5 8 9 10 17"
"3
5 0 0"
"4
0 0 0 1"
)
for e in "${edges[@]}"; do
    total=$((total+1))
    echo "$e" > /tmp/case_$$.txt
    a=$(./sol < /tmp/case_$$.txt)
    b=$(python3 brute.py < /tmp/case_$$.txt)
    if [ "$a" != "$b" ]; then
        echo "MISMATCH on edge case:"; cat /tmp/case_$$.txt; echo "sol=$a brute=$b"
        mism=$((mism+1))
    fi
done
# Random cases (small n so brute composition enumeration is feasible)
for seed in $(seq 1 $N_RAND); do
    total=$((total+1))
    mode_idx=$((seed % 4))
    case $mode_idx in
      0) mode=tiny;; 1) mode=small;; 2) mode=mid;; 3) mode=rand;;
    esac
    python3 gen.py $seed $mode > /tmp/case_$$.txt
    a=$(./sol < /tmp/case_$$.txt)
    b=$(python3 brute.py < /tmp/case_$$.txt)
    if [ "$a" != "$b" ]; then
        echo "MISMATCH seed=$seed mode=$mode:"; cat /tmp/case_$$.txt; echo "sol=$a brute=$b"
        mism=$((mism+1))
        if [ $mism -ge 5 ]; then echo "stopping after 5 mismatches"; break; fi
    fi
done
rm -f /tmp/case_$$.txt
echo "TOTAL=$total MISMATCHES=$mism"
