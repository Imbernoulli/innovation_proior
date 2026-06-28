#!/bin/bash
# Self-verify harness: seeds 1..20, compare solver to trivial all-sites baseline.
set -e
D=/srv/home/bohanlyu/innovation_proior/data_v4/ale-30/verify
LO=${1:-1}
HI=${2:-20}
sol_acc=0; base_acc=0; greedy_acc=0; nseeds=0; infeasible=0; beats=0
printf "%-5s %-7s %-7s %-7s %-10s %-10s\n" "seed" "solK" "greedK" "allK" "sol_sc" "all_sc"
for s in $(seq $LO $HI); do
  python3 "$D/gen.py" "$s" > "$D/inst_$s.txt"
  python3 "$D/baseline_all.py" "$D/inst_$s.txt" > "$D/base_$s.txt"
  "$D/sol" < "$D/inst_$s.txt" > "$D/sol_$s.txt"
  solK=$(head -1 "$D/sol_$s.txt" | awk '{print $1}')
  allK=$(head -1 "$D/base_$s.txt" | awk '{print $1}')
  sol_sc=$(python3 "$D/score.py" "$D/inst_$s.txt" "$D/sol_$s.txt")
  all_sc=$(python3 "$D/score.py" "$D/inst_$s.txt" "$D/base_$s.txt")
  # greedy baseline tower count = greedK; score.py normalises greedy to 1.0,
  # so greedK = solK * sol_sc (since sol_sc = greedK/solK).
  greedK=$(python3 -c "print(round($solK*$sol_sc))")
  printf "%-5s %-7s %-7s %-7s %-10s %-10s\n" "$s" "$solK" "$greedK" "$allK" "$sol_sc" "$all_sc"
  sol_acc=$(python3 -c "print($sol_acc+$sol_sc)")
  base_acc=$(python3 -c "print($base_acc+$all_sc)")
  nseeds=$((nseeds+1))
  python3 -c "import sys; sys.exit(0 if $sol_sc>0 else 1)" || infeasible=$((infeasible+1))
  python3 -c "import sys; sys.exit(0 if $sol_sc>$all_sc+1e-9 else 1)" && beats=$((beats+1)) || true
done
echo "----"
python3 -c "print('solver mean score = %.4f' % ($sol_acc/$nseeds))"
python3 -c "print('all-sites baseline mean score = %.4f' % ($base_acc/$nseeds))"
echo "seeds=$nseeds infeasible=$infeasible solver_beats_allbaseline=$beats"
echo "(scorer normalises GREEDY max-coverage baseline to 1.0; solver sol_sc>1 means it beats greedy too)"
