#!/usr/bin/env bash
# Differential test: sol.cpp vs brute.py over many random cases + edge cases.
set -u
DIR="/srv/home/bohanlyu/innovation_proior/data_v4/fcs-st-04/verify"
BIN="/tmp/fcs-st-04_x"
g++ -O2 -std=c++17 -o "$BIN" "$DIR/sol.cpp" || { echo "COMPILE FAIL"; exit 1; }

N="${1:-600}"
fails=0
checked=0

# Explicit edge cases (note: empty string handled separately below).
edge=( "" "a" "b" "aa" "ab" "aaa" "aba" "abc" "abacaba" "racecar" \
       "aabb" "abba" "abccba" "noon" "aabaa" "xyzzyx" "zzzz" \
       "aabbccddeeff" "abacabadabacaba" "qwertyuiop" "level" \
       "abababab" "aabbaabb" "abcba" "aabaaab" )
for t in "${edge[@]}"; do
    printf '%s' "$t" > /tmp/fcs_in.txt
    o1=$(printf '%s' "$t" | "$BIN")
    o2=$(printf '%s' "$t" | python3 "$DIR/brute.py")
    checked=$((checked+1))
    if [ "$o1" != "$o2" ]; then
        echo "MISMATCH (edge) on '$t': sol=$o1 brute=$o2"
        fails=$((fails+1))
    fi
done

for ((seed=0; seed<N; seed++)); do
    python3 "$DIR/gen.py" "$seed" > /tmp/fcs_in.txt
    o1=$("$BIN" < /tmp/fcs_in.txt)
    o2=$(python3 "$DIR/brute.py" < /tmp/fcs_in.txt)
    checked=$((checked+1))
    if [ "$o1" != "$o2" ]; then
        echo "MISMATCH (seed=$seed) input='$(cat /tmp/fcs_in.txt)': sol=$o1 brute=$o2"
        fails=$((fails+1))
        if [ "$fails" -ge 10 ]; then echo "...stopping after 10 mismatches"; break; fi
    fi
done

echo "checked=$checked fails=$fails"
if [ "$fails" -eq 0 ]; then echo "ALL PASS"; fi
