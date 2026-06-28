#!/usr/bin/env bash
# Self-verification harness for ALE-10 Wall Painting.
# Compiles the solver, runs it on seeds 1..20, scores it and two trivial
# baselines (empty Q=0, and full-canvas most-common-colour fill), and reports
# per-seed feasibility plus the means. Exit non-zero if any solver output is
# infeasible (score 0) or the solver mean does not strictly beat both baselines.
set -euo pipefail
cd "$(dirname "$0")"

g++ -O2 -std=c++17 sol.cpp -o sol

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

sum_sol=0; sum_flat=0; sum_empty=0
n=0
fail=0
printf "%-6s %8s %8s %8s %8s\n" seed sol flat empty cells
for s in $(seq 1 20); do
    python3 gen.py "$s" > "$WORK/inst.txt"
    N=$(head -1 "$WORK/inst.txt" | awk '{print $1}')
    cells=$((N*N))
    ./sol < "$WORK/inst.txt" > "$WORK/out_sol.txt"
    python3 baseline.py < "$WORK/inst.txt" > "$WORK/out_flat.txt"
    printf "0\n" > "$WORK/out_empty.txt"
    sc=$(python3 score.py "$WORK/inst.txt" "$WORK/out_sol.txt")
    fl=$(python3 score.py "$WORK/inst.txt" "$WORK/out_flat.txt")
    em=$(python3 score.py "$WORK/inst.txt" "$WORK/out_empty.txt")
    printf "%-6s %8s %8s %8s %8s\n" "$s" "$sc" "$fl" "$em" "$cells"
    if [ "$sc" -le 0 ]; then echo "  !! seed $s solver INFEASIBLE (score 0)"; fail=1; fi
    if [ "$sc" -lt "$fl" ]; then echo "  !! seed $s solver below flat baseline"; fail=1; fi
    sum_sol=$((sum_sol+sc)); sum_flat=$((sum_flat+fl)); sum_empty=$((sum_empty+em)); n=$((n+1))
done
echo "----"
echo "mean solver = $(python3 -c "print($sum_sol/$n)")"
echo "mean flat   = $(python3 -c "print($sum_flat/$n)")"
echo "mean empty  = $(python3 -c "print($sum_empty/$n)")"
if [ "$sum_sol" -le "$sum_flat" ] || [ "$sum_sol" -le "$sum_empty" ]; then
    echo "FAIL: solver mean does not strictly beat both baselines"; fail=1
fi
if [ "$fail" -ne 0 ]; then echo "SELFCHECK FAILED"; exit 1; fi
echo "SELFCHECK PASSED"
