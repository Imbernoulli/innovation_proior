#!/usr/bin/env bash
cd "$(dirname "$0")"
SOL=/tmp/cpv4-two-pointer-boundary_sol
TMP=/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad
mkdir -p "$TMP"
IN="$TMP/cpv4_in.txt"
mismatch=0
total=0
N=${1:-400}
for s in $(seq 1 "$N"); do
    python3 gen.py "$s" > "$IN"
    out_sol=$("$SOL" < "$IN")
    out_bru=$(python3 brute.py < "$IN")
    total=$((total+1))
    if [ "$out_sol" != "$out_bru" ]; then
        mismatch=$((mismatch+1))
        if [ "$mismatch" -le 8 ]; then
            echo "MISMATCH seed=$s"
            cat "$IN"
            echo "sol=$out_sol bru=$out_bru"
            echo "---"
        fi
    fi
done
echo "TOTAL=$total MISMATCHES=$mismatch"
