#!/usr/bin/env bash
# Self-verify harness for ale-50 (Adaptive Auction Bidding). Compiles sol, runs
# over a seed range, scores each, checks feasibility (a budget breach or malformed
# stream -> score 0), and compares the solver's mean normalized score against the
# spend-evenly fixed-fraction baseline (normalized to 1000000).
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sol sol.cpp
LO=${1:-1}; HI=${2:-20}
WORK=/tmp/ale50; mkdir -p "$WORK"

tot=0; n=0; wins=0; infeas=0
printf "%-5s %-12s %-10s %-12s %-12s\n" seed score feasible solver_raw base_raw
for s in $(seq "$LO" "$HI"); do
  python3 gen.py "$s" > "$WORK/inst_$s.txt"
  ./sol < "$WORK/inst_$s.txt" > "$WORK/out_$s.sol"
  sc=$(python3 score.py "$WORK/inst_$s.txt" "$WORK/out_$s.sol")
  raw=$(python3 score.py "$WORK/inst_$s.txt" "$WORK/out_$s.sol" --raw)
  # base_raw: recompute the baseline utility via a python shim.
  braw=$(python3 - "$WORK/inst_$s.txt" <<'PY'
import sys, score
T,B,value,hint,price = score.read_instance(sys.argv[1])
print(score.baseline_utility(T,B,value,hint,price))
PY
)
  # feasibility: re-derive directly from the scorer (parse + replay budget check).
  feas=$(python3 - "$WORK/inst_$s.txt" "$WORK/out_$s.sol" <<'PY'
import sys, score
T,B,value,hint,price = score.read_instance(sys.argv[1])
bids = score.read_solution(sys.argv[2], T)
if bids is None:
    print("INFEASIBLE")
else:
    u = score.replay(T,B,value,price,bids)
    print("INFEASIBLE" if u is None else "YES")
PY
)
  [ "$feas" = "INFEASIBLE" ] && infeas=$((infeas+1))
  [ "$sc" -gt 1000000 ] 2>/dev/null && wins=$((wins+1)) || true
  tot=$((tot+sc)); n=$((n+1))
  printf "%-5s %-12s %-10s %-12s %-12s\n" "$s" "$sc" "$feas" "$raw" "$braw"
done
echo "----"
echo "mean solver score = $((tot/n))   baseline=1000000   wins=$wins/$n   parse_infeasible=$infeas/$n"
