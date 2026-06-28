#!/bin/bash
# Self-verify harness for ale-33: seeds LO..HI.
# Confirms every solver output is FEASIBLE (score>0) and the solver's mean score
# strictly beats the trivial greedy baseline (sequential hard-blocked routing).
set -e
D=/srv/home/bohanlyu/innovation_proior/data_v4/ale-33/verify
LO=${1:-1}
HI=${2:-20}
sol_acc=0; base_acc=0; nseeds=0; infeasible=0; beats=0; baseinfeas=0
printf "%-5s %-8s %-10s %-10s\n" "seed" "n" "sol_sc" "greedy_sc"
for s in $(seq $LO $HI); do
  python3 "$D/gen.py" "$s" > "$D/inst_$s.txt"
  "$D/sol"    < "$D/inst_$s.txt" > "$D/sol_$s.txt"
  "$D/greedy" < "$D/inst_$s.txt" > "$D/grd_$s.txt"
  n=$(head -1 "$D/inst_$s.txt" | awk '{print $3}')
  sol_sc=$(python3 "$D/score.py" "$D/inst_$s.txt" "$D/sol_$s.txt")
  grd_sc=$(python3 "$D/score.py" "$D/inst_$s.txt" "$D/grd_$s.txt")
  printf "%-5s %-8s %-10s %-10s\n" "$s" "$n" "$sol_sc" "$grd_sc"
  sol_acc=$(python3 -c "print($sol_acc+$sol_sc)")
  base_acc=$(python3 -c "print($base_acc+$grd_sc)")
  nseeds=$((nseeds+1))
  python3 -c "import sys; sys.exit(0 if $sol_sc>0 else 1)" || infeasible=$((infeasible+1))
  python3 -c "import sys; sys.exit(0 if $grd_sc<=0 else 1)" && baseinfeas=$((baseinfeas+1)) || true
  python3 -c "import sys; sys.exit(0 if $sol_sc>$grd_sc+1e-9 else 1)" && beats=$((beats+1)) || true
done
echo "----"
python3 -c "print('solver mean score        = %.4f' % ($sol_acc/$nseeds))"
python3 -c "print('greedy baseline mean score= %.4f' % ($base_acc/$nseeds))"
echo "seeds=$nseeds  solver_infeasible=$infeasible  greedy_infeasible_seeds=$baseinfeas  solver_beats_greedy=$beats"
