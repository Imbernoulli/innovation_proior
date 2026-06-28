#!/usr/bin/env bash
# Self-verify harness for ale-42. Compiles sol, runs over a seed range, scores
# each, checks feasibility (a feasibility failure makes BOTH score and raw 0 with
# a clearly-empty report being the only legitimate 0), and compares the solver's
# mean normalized score against the uniform-grid baseline (==1000000).
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sol sol.cpp
LO=${1:-1}; HI=${2:-20}
WORK=/tmp/ale42; mkdir -p "$WORK"
printf "0\n0\n" > "$WORK/empty.sol"   # trivial baseline: empty report

tot=0; n=0; wins=0; infeas=0
printf "%-5s %-12s %-10s %-12s %-12s\n" seed score feasible solver_raw base_raw
for s in $(seq "$LO" "$HI"); do
  python3 gen.py "$s" > "$WORK/inst_$s.txt"
  ./sol < "$WORK/inst_$s.txt" > "$WORK/out_$s.sol"
  sc=$(python3 score.py "$WORK/inst_$s.txt" "$WORK/out_$s.sol")
  raw=$(python3 score.py "$WORK/inst_$s.txt" "$WORK/out_$s.sol" --raw)
  # base_raw: recompute baseline value with a tiny python shim
  braw=$(python3 - "$WORK/inst_$s.txt" <<'PY'
import sys, score
H,W,Q,rad,sigma,thr,pen,grid = score.read_instance(sys.argv[1])
print(score.baseline_value(H,W,Q,rad,thr,pen,grid))
PY
)
  # feasibility: the scorer returns None -> we cannot tell from score alone, so
  # re-derive: an empty report is the ONLY legit way to get value 0. Detect a
  # parse/feasibility failure by checking the report is well-formed via --raw on
  # a guaranteed-feasible re-emit is overkill; instead trust that score>0 OR an
  # empty report => feasible. We flag INFEASIBLE only if the python scorer says
  # the solution failed to parse, which we test directly:
  feas=$(python3 - "$WORK/inst_$s.txt" "$WORK/out_$s.sol" <<'PY'
import sys, score
H,W,Q,rad,sigma,thr,pen,grid = score.read_instance(sys.argv[1])
print("YES" if score.read_solution(sys.argv[2],H,W,Q,rad,grid) is not None else "INFEASIBLE")
PY
)
  [ "$feas" = "INFEASIBLE" ] && infeas=$((infeas+1))
  [ "$sc" -gt 1000000 ] 2>/dev/null && wins=$((wins+1)) || true
  tot=$((tot+sc)); n=$((n+1))
  printf "%-5s %-12s %-10s %-12s %-12s\n" "$s" "$sc" "$feas" "$raw" "$braw"
done
echo "----"
echo "mean solver score = $((tot/n))   baseline=1000000   wins=$wins/$n   parse_infeasible=$infeas/$n"
