#!/usr/bin/env bash
set -u
SP="$1"
N="$2"
DIR="/srv/home/bohanlyu/innovation_proior/data_v4/cpv4-strings-hash-negzero/verify"
SOL="/tmp/cpv4-strings-hash-negzero_sol"
mism=0
total=0
for s in $(seq 1 "$N"); do
    f="$SP/case_$s.txt"
    python3 "$DIR/gen.py" "$s" > "$f"
    b=$(python3 "$DIR/brute.py" < "$f")
    o=$("$SOL" < "$f")
    total=$((total+1))
    if [ "$b" != "$o" ]; then
        mism=$((mism+1))
        echo "MISMATCH seed=$s brute=$b sol=$o"
        cat "$f"
        if [ "$mism" -ge 8 ]; then break; fi
    fi
    rm -f "$f"
done
echo "TOTAL=$total MISMATCHES=$mism"
