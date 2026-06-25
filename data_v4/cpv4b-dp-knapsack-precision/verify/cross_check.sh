#!/usr/bin/env bash
set -u
DIR="/srv/home/bohanlyu/innovation_proior/data_v4/cpv4b-dp-knapsack-precision/verify"
SD="/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad"
mkdir -p "$SD"
# 1) oracle_dp vs brute on small gen cases (trust the DP oracle)
bad=0
for i in $(seq 1 300); do
  python3 "$DIR/gen.py" "$i" > "$SD/cc.txt"
  o=$(python3 "$DIR/oracle_dp.py" < "$SD/cc.txt")
  b=$(python3 "$DIR/brute.py" < "$SD/cc.txt")
  if [ "$o" != "$b" ]; then echo "ORACLE!=BRUTE seed=$i  oracle=$o brute=$b"; cat "$SD/cc.txt"; bad=$((bad+1)); [ $bad -ge 3 ] && break; fi
done
echo "ORACLE_VS_BRUTE_MISMATCHES=$bad"
rm -f "$SD/cc.txt"
