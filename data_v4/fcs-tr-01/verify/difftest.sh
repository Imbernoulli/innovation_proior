#!/usr/bin/env bash
# Differential test: sol vs brute. Usage: difftest.sh <start> <end> <maxn> <maxq>
set -u
cd "$(dirname "$0")"
S=${1:-1}; E=${2:-600}; MN=${3:-14}; MQ=${4:-6}
mism=0; err=0
for s in $(seq "$S" "$E"); do
  python3 gen.py "$s" "$MN" "$MQ" > /tmp/dtin.txt 2>/tmp/dterr.txt
  o1=$(/tmp/fcs-tr-01_x < /tmp/dtin.txt)
  o2=$(python3 brute.py < /tmp/dtin.txt 2>/tmp/dperr.txt)
  if [ -s /tmp/dperr.txt ]; then
    echo "BRUTE ERR seed=$s"; cat /tmp/dtin.txt; cat /tmp/dperr.txt; err=$((err+1)); break
  fi
  if [ "$o1" != "$o2" ]; then
    echo "MISMATCH seed=$s"; cat /tmp/dtin.txt
    echo "--sol:"; echo "$o1"; echo "--brute:"; echo "$o2"
    mism=$((mism+1)); [ $mism -ge 3 ] && break
  fi
done
echo "mismatches=$mism errors=$err over [$S,$E]"
