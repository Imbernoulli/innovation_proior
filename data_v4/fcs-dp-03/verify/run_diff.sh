#!/usr/bin/env bash
# Differential test: sol.cpp vs brute.py over many random cases + explicit edges.
set -u
cd "$(dirname "$0")/.."
BIN=/tmp/fcs-dp-03_x
g++ -O2 -std=c++17 -o "$BIN" verify/sol.cpp || { echo "COMPILE FAIL"; exit 1; }

N=${1:-600}
mismatch=0
for ((s=0; s<N; s++)); do
  python3 verify/gen.py "$s" > /tmp/fcs_dp03_in.txt
  out_sol=$("$BIN" < /tmp/fcs_dp03_in.txt)
  out_bru=$(python3 verify/brute.py < /tmp/fcs_dp03_in.txt)
  if [[ "$out_sol" != "$out_bru" ]]; then
    echo "MISMATCH seed=$s sol=$out_sol brute=$out_bru"
    echo "--- input ---"; cat /tmp/fcs_dp03_in.txt
    mismatch=$((mismatch+1))
    if [[ $mismatch -ge 5 ]]; then break; fi
  fi
done

echo "--- explicit edge cases ---"
edge_check() {
  local desc="$1"; local inp="$2"
  local a; a=$(printf "%b" "$inp" | "$BIN")
  local b; b=$(printf "%b" "$inp" | python3 verify/brute.py)
  if [[ "$a" != "$b" ]]; then echo "EDGE MISMATCH [$desc] sol=$a brute=$b inp=$inp"; mismatch=$((mismatch+1));
  else echo "ok [$desc] -> $a"; fi
}
edge_check "n=0"            "0\n"
edge_check "n=1"           "1\n5\n"
edge_check "n=2 equal"     "2\n7 7\n"
edge_check "n=2 big"       "2\n1000000000 1000000000\n"
edge_check "ascending"     "4\n1 2 3 4\n"
edge_check "descending"    "4\n4 3 2 1\n"
edge_check "all-equal-5"   "5\n3 3 3 3 3\n"
edge_check "all-equal-6"   "6\n1 1 1 1 1 1\n"
edge_check "big-uniform"   "8\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000\n"
edge_check "spiky"         "7\n1 1000000000 1 1000000000 1 1000000000 1\n"

echo "==== mismatches: $mismatch ===="
