#!/usr/bin/env bash
set -euo pipefail
DIR="/srv/home/bohanlyu/innovation_proior/data_v4/ale-v2-01/verify"
WORK="${1:-/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad/alev201}"
mkdir -p "$WORK"
g++ -O2 -std=c++17 -o "$WORK/sol" "$DIR/sol.cpp"
echo "compiled OK"

sum_sol=0; sum_triv=0; sum_sav=0
nfeas=0; nbeat_triv=0; nbeat_sav=0; total=0
for seed in $(seq 1 20); do
  inst="$WORK/inst_$seed.txt"
  python3 "$DIR/gen.py" "$seed" > "$inst"
  "$WORK/sol" < "$inst" > "$WORK/sol_$seed.txt"
  python3 "$DIR/baseline_trivial.py" < "$inst" > "$WORK/triv_$seed.txt"
  python3 "$DIR/baseline_savings.py" < "$inst" > "$WORK/sav_$seed.txt"
  s_sol=$(python3 "$DIR/score.py" "$inst" "$WORK/sol_$seed.txt" 2> "$WORK/sol_$seed.err")
  s_triv=$(python3 "$DIR/score.py" "$inst" "$WORK/triv_$seed.txt" 2>/dev/null)
  s_sav=$(python3 "$DIR/score.py" "$inst" "$WORK/sav_$seed.txt" 2>/dev/null)
  len_sol=$(grep -o 'total_length=[0-9.]*' "$WORK/sol_$seed.err" | cut -d= -f2 || echo NA)
  total=$((total+1))
  awk -v a="$s_sol" 'BEGIN{exit !(a>0)}' && nfeas=$((nfeas+1))
  awk -v a="$s_sol" -v b="$s_triv" 'BEGIN{exit !(a>b)}' && nbeat_triv=$((nbeat_triv+1))
  awk -v a="$s_sol" -v b="$s_sav" 'BEGIN{exit !(a>=b)}' && nbeat_sav=$((nbeat_sav+1))
  sum_sol=$(awk -v s="$sum_sol" -v v="$s_sol" 'BEGIN{print s+v}')
  sum_triv=$(awk -v s="$sum_triv" -v v="$s_triv" 'BEGIN{print s+v}')
  sum_sav=$(awk -v s="$sum_sav" -v v="$s_sav" 'BEGIN{print s+v}')
  printf "seed %2d  sol=%10s triv=%10s sav=%10s  len_sol=%s\n" "$seed" "$s_sol" "$s_triv" "$s_sav" "$len_sol"
done
echo "----"
echo "feasible: $nfeas/$total   beat_trivial: $nbeat_triv/$total   >=savings: $nbeat_sav/$total"
awk -v a="$sum_sol" -v b="$sum_triv" -v c="$sum_sav" -v n="$total" \
  'BEGIN{printf "mean sol=%.4f  mean triv=%.4f  mean sav=%.4f\n", a/n, b/n, c/n}'
