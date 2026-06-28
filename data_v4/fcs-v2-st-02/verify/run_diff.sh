#!/usr/bin/env bash
set -e
EXE=/tmp/fcs-v2-st-02_x
DIR=/srv/home/bohanlyu/innovation_proior/data_v4/fcs-v2-st-02/verify
N=${1:-500}
mism=0
for i in $(seq 1 "$N"); do
  python3 "$DIR/gen.py" "$i" > /tmp/fcs_in.txt
  out_sol=$("$EXE" < /tmp/fcs_in.txt)
  out_bru=$(python3 "$DIR/brute.py" < /tmp/fcs_in.txt)
  if [ "$out_sol" != "$out_bru" ]; then
    echo "MISMATCH seed=$i  sol=$out_sol  brute=$out_bru  input=$(cat /tmp/fcs_in.txt)"
    mism=$((mism+1))
    if [ "$mism" -ge 10 ]; then break; fi
  fi
done
echo "done $N cases, mismatches=$mism"
