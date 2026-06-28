#!/bin/bash
declare -a cases=(
"0"
"1
-7"
"1
5"
"2
3 4"
"2
-3 -4"
"3
5 5 5"
"3
-1 -2 -3"
"4
8 9 2 9"
"5
1 2 3 4 5"
"6
5 1 1 5 1 5"
"3
1000000000 1000000000 1000000000"
"4
1000000000 1000000000 1000000000 1000000000"
"7
8 9 2 9 9 -2 8"
"8
8 9 2 9 9 -2 8 -5"
"5
-1 0 -1 0 -1"
"6
0 0 0 0 0 0"
"4
-5 10 -5 10"
)
mm=0
for c in "${cases[@]}"; do
  echo "$c" > e.txt
  s=$(./sol < e.txt); b=$(python3 brute.py < e.txt)
  if [ "$s" != "$b" ]; then echo "EDGE MISMATCH: input=[$c] sol=$s brute=$b"; mm=$((mm+1)); else echo "ok ($s) <- $(echo $c | tr '\n' ' ')"; fi
done
echo "EDGE mismatches=$mm"
