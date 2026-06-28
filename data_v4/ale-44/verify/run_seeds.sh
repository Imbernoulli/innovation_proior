#!/usr/bin/env bash
# Self-verify harness: for seeds 1..N, run sol + baseline, score both,
# confirm every sol output is feasible (score>0 parse ok) and sol mean > baseline mean.
set -e
cd "$(dirname "$0")"
N="${1:-20}"
g++ -O2 -std=c++17 -o sol sol.cpp
WORK="$(mktemp -d)"
sol_sum=0; base_sum=0; feasible_ok=1; beat_cnt=0; total=0
printf "%5s | %8s | %8s | %6s\n" seed sol base nS
for s in $(seq 1 "$N"); do
    python3 gen.py "$s" > "$WORK/inst.txt"
    ./sol < "$WORK/inst.txt" > "$WORK/sol.txt"
    python3 baseline.py < "$WORK/inst.txt" > "$WORK/base.txt"
    sc=$(python3 score.py "$WORK/inst.txt" "$WORK/sol.txt")
    bc=$(python3 score.py "$WORK/inst.txt" "$WORK/base.txt")
    nS=$(sed -n '2p' "$WORK/inst.txt" | awk '{print $1}')
    printf "%5s | %8s | %8s | %6s\n" "$s" "$sc" "$bc" "$nS"
    sol_sum=$((sol_sum + sc)); base_sum=$((base_sum + bc)); total=$((total+1))
    if [ "$sc" -le 0 ]; then feasible_ok=0; fi
    if [ "$sc" -gt "$bc" ]; then beat_cnt=$((beat_cnt+1)); fi
done
rm -rf "$WORK"
echo "----"
echo "sol_sum=$sol_sum base_sum=$base_sum (over $total seeds)"
echo "sol_mean=$(python3 -c "print($sol_sum/$total)") base_mean=$(python3 -c "print($base_sum/$total)")"
echo "all_feasible(score>0)=$feasible_ok  sol_strictly_beats_base_on_count=$beat_cnt/$total"
echo "MEAN_BEATS=$(python3 -c "print(1 if $sol_sum>$base_sum else 0)")"
