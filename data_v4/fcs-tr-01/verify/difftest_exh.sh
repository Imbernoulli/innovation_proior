#!/usr/bin/env bash
# sol vs the truly-independent EXHAUSTIVE oracle on tiny cases.
set -u
cd "$(dirname "$0")"
S=${1:-1}; E=${2:-300}; MN=${3:-8}; MQ=${4:-4}
mism=0
for s in $(seq "$S" "$E"); do
  python3 gen.py "$s" "$MN" "$MQ" > /tmp/dtin.txt
  o1=$(/tmp/fcs-tr-01_x < /tmp/dtin.txt)
  o2=$(python3 brute_exhaustive.py < /tmp/dtin.txt)
  if [ "$o1" != "$o2" ]; then
    echo "MISMATCH seed=$s"; cat /tmp/dtin.txt
    echo "--sol:"; echo "$o1"; echo "--exh:"; echo "$o2"
    mism=$((mism+1)); [ $mism -ge 3 ] && break
  fi
done
echo "exh-mismatches=$mism over [$S,$E]"
