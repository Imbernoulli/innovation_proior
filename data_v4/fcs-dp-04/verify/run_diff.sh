#!/usr/bin/env bash
set -e
DIR=/srv/home/bohanlyu/innovation_proior/data_v4/fcs-dp-04
SOL=/tmp/fcs-dp-04_x
g++ -O2 -std=c++17 -o "$SOL" "$DIR/verify/sol.cpp"

N=${1:-600}
mismatch=0
for ((s=0; s<N; s++)); do
  python3 "$DIR/verify/gen.py" "$s" > /tmp/fcs04_in.txt
  a=$("$SOL" < /tmp/fcs04_in.txt)
  b=$(python3 "$DIR/verify/brute.py" < /tmp/fcs04_in.txt)
  if [ "$a" != "$b" ]; then
    echo "MISMATCH seed=$s sol=$a brute=$b"
    cat /tmp/fcs04_in.txt
    echo "---"
    mismatch=$((mismatch+1))
    if [ "$mismatch" -ge 10 ]; then break; fi
  fi
done
echo "Done. random cases=$N mismatches=$mismatch"
