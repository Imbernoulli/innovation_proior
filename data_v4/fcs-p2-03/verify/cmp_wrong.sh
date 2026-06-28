#!/bin/bash
D=/srv/home/bohanlyu/innovation_proior/data_v4/fcs-p2-03/verify
cd "$D"
run() {
  printf '%s\n' "$1" > _c.txt
  echo "input=[$(tr '\n' ' ' < _c.txt | sed 's/  */ /g')] brute=$(python3 brute.py<_c.txt) wrongKadane=$(python3 wrong_kadane.py<_c.txt) sol=$(./sol<_c.txt)"
}
run "4
2 3 -2 4"
run "2
-2 -3"
run "5
-1 -2 -3 -4 -5"
run "3
-2 3 -4"
run "6
2 -5 -2 -4 3 -1"
rm -f _c.txt
