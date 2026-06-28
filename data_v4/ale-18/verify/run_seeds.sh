#!/usr/bin/env bash
# Self-verification harness: compile, then over seeds 1..20 run the solver and
# a trivial baseline (identity permutation 1..n), score both, confirm every
# solver output is feasible (score > 0) and the solver mean strictly beats the
# baseline mean.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sol sol.cpp

SUM_SOL=0; SUM_BASE=0; CNT=0; ALL_FEAS=1
printf "seed |  n  | lambda |  solver_score | baseline(EDF) | base_id_score\n"
for s in $(seq 1 20); do
  python3 gen.py "$s" > /tmp/ale18_i_$s.txt
  n=$(head -1 /tmp/ale18_i_$s.txt | awk '{print $1}')
  lam=$(head -1 /tmp/ale18_i_$s.txt | awk '{print $2}')
  ./sol < /tmp/ale18_i_$s.txt > /tmp/ale18_o_$s.txt
  # trivial baseline 1: identity permutation 1..n
  python3 - "$n" > /tmp/ale18_id_$s.txt <<'PY'
import sys
n=int(sys.argv[1])
print(' '.join(str(i) for i in range(1,n+1)))
PY
  sc=$(python3 score.py /tmp/ale18_i_$s.txt /tmp/ale18_o_$s.txt)
  idsc=$(python3 score.py /tmp/ale18_i_$s.txt /tmp/ale18_id_$s.txt)
  # EDF baseline score is 1.0 by construction (it is the normalizer)
  feas=$(python3 -c "print(1 if float('$sc')>0 else 0)")
  if [ "$feas" != "1" ]; then ALL_FEAS=0; fi
  printf "%4s | %3s | %6s | %13s | %13s | %12s\n" "$s" "$n" "$lam" "$sc" "1.0" "$idsc"
  SUM_SOL=$(python3 -c "print($SUM_SOL + float('$sc'))")
  SUM_BASE=$(python3 -c "print($SUM_BASE + float('$idsc'))")
  CNT=$((CNT+1))
done
echo "-------------------------------------------------------------------"
python3 -c "print('solver mean   =', $SUM_SOL/$CNT)"
python3 -c "print('id baseline   =', $SUM_BASE/$CNT)"
echo "EDF baseline mean = 1.0 (normalizer)"
echo "all_feasible = $ALL_FEAS"
python3 -c "print('beats_EDF      =', $SUM_SOL/$CNT > 1.0)"
python3 -c "print('beats_id_base  =', $SUM_SOL/$CNT > $SUM_BASE/$CNT)"
