#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
SOL=/tmp/cpv4b-two-pointer-negzero_sol
TMP=/tmp/cpv4b_stress
mkdir -p "$TMP"
fails=0
N=${1:-500}
for s in $(seq 1 "$N"); do
  in="$TMP/in_$s.txt"
  python3 gen.py "$s" > "$in"
  os=$("$SOL" < "$in")
  ob=$(python3 brute.py < "$in")
  if [ "$os" != "$ob" ]; then
    echo "MISMATCH seed=$s"
    cat "$in"
    echo "sol=[$os] brute=[$ob]"
    fails=$((fails+1))
    if [ "$fails" -ge 8 ]; then break; fi
  fi
done
echo "TOTAL_FAILS=$fails over $N seeds"
