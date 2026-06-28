#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o sol sol.cpp 2>/dev/null || { echo "COMPILE FAIL"; exit 1; }
sumS=0; sumB=0; feasfail=0; lose=0
NSEEDS=${1:-20}
for s in $(seq 1 "$NSEEDS"); do
  python3 gen.py "$s" > inst_$s.txt
  ./sol < inst_$s.txt > out_sol_$s.txt
  python3 baseline.py < inst_$s.txt > out_base_$s.txt
  ss=$(python3 score.py inst_$s.txt out_sol_$s.txt)
  bs=$(python3 score.py inst_$s.txt out_base_$s.txt)
  hdr=$(head -1 inst_$s.txt)
  G=$(echo "$hdr" | cut -d' ' -f1); M=$(echo "$hdr" | cut -d' ' -f2); Bb=$(echo "$hdr" | cut -d' ' -f3)
  if [ "$ss" -le 0 ]; then feasfail=$((feasfail+1)); fi
  if [ "$ss" -le "$bs" ]; then lose=$((lose+1)); fi
  sumS=$((sumS+ss)); sumB=$((sumB+bs))
  printf "seed %2d  G=%s M=%s B=%s  sol=%8d  base=%8d  delta=%+d\n" "$s" "$G" "$M" "$Bb" "$ss" "$bs" "$((ss-bs))"
done
echo "-----"
echo "mean sol=$((sumS/NSEEDS))  mean base=$((sumB/NSEEDS))  feasfail=$feasfail  lose_or_tie=$lose"
