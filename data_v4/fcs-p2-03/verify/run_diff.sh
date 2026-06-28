#!/bin/bash
set -e
D=/srv/home/bohanlyu/innovation_proior/data_v4/fcs-p2-03/verify
cd "$D"
edges=(
"1
0"
"1
9"
"1
-9"
"2
0 0"
"2
-9 -9"
"2
9 9"
"3
-1 -1 -1"
"18
9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9"
"18
-9 -9 -9 -9 -9 -9 -9 -9 -9 -9 -9 -9 -9 -9 -9 -9 -9 -9"
"5
-9 0 -9 0 -9"
"4
0 -1 0 -1"
"6
-2 -3 -2 -3 -2 -3"
"7
-1 2 -3 4 -5 6 -7"
)
fail=0
for e in "${edges[@]}"; do
  printf '%s\n' "$e" > _e.txt
  o1=$(./sol < _e.txt)
  o2=$(python3 brute.py < _e.txt)
  if [ "$o1" != "$o2" ]; then echo "EDGE MISMATCH input=[$e] sol=$o1 brute=$o2"; fail=1; fi
done
echo "edge_fail=$fail (over ${#edges[@]} edges)"
mismatch=0
N=${1:-600}
for s in $(seq 1 "$N"); do
  python3 gen.py "$s" > _t.txt
  o1=$(./sol < _t.txt)
  o2=$(python3 brute.py < _t.txt)
  if [ "$o1" != "$o2" ]; then
    echo "MISMATCH seed=$s sol=$o1 brute=$o2 input:"; cat _t.txt
    mismatch=$((mismatch+1))
    if [ $mismatch -ge 5 ]; then break; fi
  fi
done
echo "random_mismatches=$mismatch (over $N)"
rm -f _t.txt _e.txt
