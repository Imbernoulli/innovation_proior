#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sol sol.cpp

SEEDS="${1:-20}"
sumS=0; sumB=0; sumE=0; nfeas=0; total=0; minS=1e18
echo "seed |    solver    |   identity   |     EDD      | feasible"
for s in $(seq 1 "$SEEDS"); do
  python3 gen.py "$s" > inst_$s.txt
  ./sol < inst_$s.txt > sol_$s.txt
  # trivial baseline 1: identity order (input order)
  n=$(head -1 inst_$s.txt)
  seq 0 $((n-1)) > ident_$s.txt
  # trivial baseline 2: EDD (scorer's normalizer; should print ~1e6)
  # build EDD with python
  python3 - "$s" <<'PY' > edd_$s.txt
import sys
s=sys.argv[1]
toks=open(f"inst_{s}.txt").read().split()
it=iter(toks); n=int(next(it))
d=[]
for i in range(n):
    p=int(next(it)); w=int(next(it)); dj=int(next(it)); d.append((dj,i))
order=[i for _,i in sorted(d, key=lambda x:(x[0],x[1]))]
print("\n".join(map(str,order)))
PY
  sc=$(python3 score.py inst_$s.txt sol_$s.txt)
  bi=$(python3 score.py inst_$s.txt ident_$s.txt)
  be=$(python3 score.py inst_$s.txt edd_$s.txt)
  feas=$(python3 -c "print('YES' if $sc>0 else 'NO')")
  printf "%4s | %12.1f | %12.1f | %12.1f | %s\n" "$s" "$sc" "$bi" "$be" "$feas"
  sumS=$(python3 -c "print($sumS+$sc)")
  sumB=$(python3 -c "print($sumB+$bi)")
  sumE=$(python3 -c "print($sumE+$be)")
  total=$((total+1))
  if python3 -c "exit(0 if $sc>0 else 1)"; then nfeas=$((nfeas+1)); fi
  minS=$(python3 -c "print(min($minS,$sc))")
done
echo "----"
python3 -c "print(f'mean solver  = {$sumS/$total:.1f}')"
python3 -c "print(f'mean identity= {$sumB/$total:.1f}')"
python3 -c "print(f'mean EDD     = {$sumE/$total:.1f}')"
echo "feasible: $nfeas / $total"
python3 -c "print(f'min solver score = {$minS:.1f}')"
python3 -c "print('BEATS EDD baseline (mean):', $sumS/$total > $sumE/$total)"
python3 -c "print('ALL FEASIBLE:', $nfeas==$total)"
