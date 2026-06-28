#!/usr/bin/env bash
set -euo pipefail
DIR=/srv/home/bohanlyu/innovation_proior/data_v4/fcs-v2-ds-01
SOL=/tmp/fcs-v2-ds-01_x
TMP=$(mktemp -d)
N=${1:-600}
mism=0; total=0
for seed in $(seq 1 "$N"); do
  python3 "$DIR/verify/gen.py" "$seed" > "$TMP/in.txt"
  "$SOL" < "$TMP/in.txt" > "$TMP/sa.txt"
  python3 "$DIR/verify/brute.py" < "$TMP/in.txt" > "$TMP/sb.txt"
  if ! diff -q "$TMP/sa.txt" "$TMP/sb.txt" > /dev/null; then
    mism=$((mism+1))
    if [ "$mism" -le 3 ]; then
      echo "MISMATCH seed=$seed"
      echo "--- input ---"; cat "$TMP/in.txt"
      echo "--- sol ---"; cat "$TMP/sa.txt"
      echo "--- brute ---"; cat "$TMP/sb.txt"
    fi
  fi
  total=$((total+1))
done
echo "TOTAL=$total MISMATCHES=$mism"
rm -rf "$TMP"
