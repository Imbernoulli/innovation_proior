#!/usr/bin/env bash
set -u
DIR="/srv/home/bohanlyu/innovation_proior/data_v4/fcs-gx-02"
BIN="/tmp/fcs-gx-02_x"
TMP="$(mktemp)"
pass=0; fail=0; firstfail=""
N="${1:-700}"
for seed in $(seq 1 "$N"); do
  python3 "$DIR/verify/gen.py" "$seed" > "$TMP"
  out1=$("$BIN" < "$TMP")
  out2=$(python3 "$DIR/verify/brute.py" < "$TMP")
  if [ "$out1" == "$out2" ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    if [ -z "$firstfail" ]; then
      firstfail=$seed
      echo "MISMATCH seed=$seed"
      echo "INPUT:"; cat "$TMP"
      echo "SOL=[$out1] BRUTE=[$out2]"
    fi
  fi
done
rm -f "$TMP"
echo "PASS=$pass FAIL=$fail"
