#!/usr/bin/env bash
set -e
DIR=/srv/home/bohanlyu/innovation_proior/data_v4/fcs-dp-07
BIN=/tmp/fcs-dp-07_x
N=${1:-200}
mismatch=0; total=0
TMP=$(mktemp)
for seed in $(seq 1 $N); do
  python3 "$DIR/verify/gen_big.py" "$seed" > "$TMP"
  out_sol=$("$BIN" < "$TMP")
  out_bru=$(python3 "$DIR/verify/brute_big.py" < "$TMP")
  total=$((total+1))
  if [ "$out_sol" != "$out_bru" ]; then
    mismatch=$((mismatch+1))
    echo "MISMATCH seed=$seed input=[$(cat "$TMP")] sol=$out_sol big=$out_bru"
    if [ $mismatch -ge 8 ]; then break; fi
  fi
done
echo "BIG TOTAL=$total MISMATCHES=$mismatch"
rm -f "$TMP"
