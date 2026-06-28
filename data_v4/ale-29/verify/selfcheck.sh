#!/usr/bin/env bash
# Self-verify: compile, generate seeds, run sol + baseline, score both, confirm
# every output is feasible (score>0 where it should be) and sol beats baseline.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sol sol.cpp 2>/dev/null

SEEDS="${1:-20}"
sol_sum=0
base_sum=0
nfeas=0
n=0
echo "seed       sol      baseline   sol_feasible"
for s in $(seq 1 "$SEEDS"); do
    python3 gen.py "$s" > inst_$s.txt
    ./sol < inst_$s.txt > sol_$s.txt
    python3 baseline.py < inst_$s.txt > base_$s.txt
    sv=$(python3 score.py inst_$s.txt sol_$s.txt)
    bv=$(python3 score.py inst_$s.txt base_$s.txt)
    feas="YES"
    # feasibility: a non-zero score means the scorer accepted it. Also explicitly
    # re-validate with the scorer's own feasibility check via exit behavior.
    if python3 - "$sv" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > 0 else 1)
PY
    then nfeas=$((nfeas+1)); else feas="NO(score=0)"; fi
    printf "%-4s  %10.1f  %10.1f   %s\n" "$s" "$sv" "$bv" "$feas"
    sol_sum=$(python3 -c "print($sol_sum + $sv)")
    base_sum=$(python3 -c "print($base_sum + $bv)")
    n=$((n+1))
    rm -f inst_$s.txt sol_$s.txt base_$s.txt
done
echo "----"
python3 -c "
n=$n
ss=$sol_sum; bs=$base_sum
print(f'mean sol      = {ss/n:.2f}')
print(f'mean baseline = {bs/n:.2f}')
print(f'feasible      = $nfeas / {n}')
print(f'sol beats baseline: {\"YES\" if ss>bs else \"NO\"}  (ratio {ss/max(bs,1e-9):.4f})')
"
