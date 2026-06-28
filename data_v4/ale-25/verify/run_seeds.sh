#!/usr/bin/env bash
# Self-verify harness: generate seeds 1..N, run solver, score, compare to the
# trivial first-fit-by-start baseline. Confirms every output is feasible
# (raw>0) and the solver's mean raw score strictly beats the baseline mean.
set -euo pipefail
cd "$(dirname "$0")"
N="${1:-20}"

g++ -O2 -std=c++17 -o sol sol.cpp

sum_sol=0
sum_base=0
cnt=0
infeasible=0
worse=0

printf "%5s %12s %12s %10s\n" "seed" "solver_raw" "base_raw" "norm"
for s in $(seq 1 "$N"); do
  python3 gen.py "$s" > "inst_$s.txt"
  ./sol < "inst_$s.txt" > "sol_$s.txt"
  raw=$(python3 score.py "inst_$s.txt" "sol_$s.txt" --raw)
  base=$(python3 - "inst_$s.txt" <<'PY'
import sys
sys.argv=['x', sys.argv[1]]
import importlib.util
spec=importlib.util.spec_from_file_location("score","score.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
n,K,iv=m.read_instance(sys.argv[1])
print(m.first_fit_baseline(n,K,iv))
PY
)
  norm=$(python3 score.py "inst_$s.txt" "sol_$s.txt")
  printf "%5s %12s %12s %10s\n" "$s" "$raw" "$base" "$norm"
  if [ "$raw" -le 0 ]; then infeasible=$((infeasible+1)); fi
  if [ "$raw" -le "$base" ]; then worse=$((worse+1)); fi
  sum_sol=$((sum_sol+raw))
  sum_base=$((sum_base+base))
  cnt=$((cnt+1))
done

echo "-----------------------------------------------"
echo "seeds=$cnt  infeasible=$infeasible  solver<=baseline=$worse"
python3 - "$sum_sol" "$sum_base" "$cnt" <<'PY'
import sys
ss,sb,c=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
print(f"mean solver = {ss/c:.1f}   mean baseline = {sb/c:.1f}   ratio = {ss/sb:.4f}")
PY
