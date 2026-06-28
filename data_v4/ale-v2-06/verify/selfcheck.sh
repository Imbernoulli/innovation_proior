#!/usr/bin/env bash
# Self-verification: generate seeds 1..20, run sol, score it, compare to baselines.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sol sol.cpp 2>/dev/null
PY=python3
TMP=$(mktemp -d)
sum_sol=0; sum_empty=0; nfeas=0; nbeat=0; total=0
echo "seed  sol_score  empty_score  feasible  beats_baseline(1e6)"
for s in $(seq 1 20); do
  $PY gen.py "$s" > "$TMP/inst.txt"
  ./sol < "$TMP/inst.txt" > "$TMP/out.txt"
  sc=$($PY score.py "$TMP/inst.txt" "$TMP/out.txt")
  # empty-set baseline
  printf "0\n" > "$TMP/empty.txt"
  es=$($PY score.py "$TMP/inst.txt" "$TMP/empty.txt")
  feas="yes"; [ "$sc" -le 0 ] && feas="NO"
  beat="yes"; [ "$sc" -le 1000000 ] && beat="NO"
  [ "$feas" = "yes" ] && nfeas=$((nfeas+1))
  [ "$beat" = "yes" ] && nbeat=$((nbeat+1))
  sum_sol=$((sum_sol+sc)); sum_empty=$((sum_empty+es)); total=$((total+1))
  printf "%4s  %9s  %11s  %8s  %s\n" "$s" "$sc" "$es" "$feas" "$beat"
done
echo "---"
echo "mean sol  = $((sum_sol/total))"
echo "mean empty= $((sum_empty/total))"
echo "feasible  : $nfeas / $total"
echo "beats 1e6 : $nbeat / $total"
rm -rf "$TMP"
