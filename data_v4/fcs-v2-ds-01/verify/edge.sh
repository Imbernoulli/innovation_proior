#!/usr/bin/env bash
set -euo pipefail
SOL=/tmp/fcs-v2-ds-01_x
TMP=$(mktemp -d)

check() {
  local name="$1"; local input="$2"; local expect="$3"
  printf '%s' "$input" > "$TMP/in.txt"
  got=$("$SOL" < "$TMP/in.txt" | tr '\n' ' ' | sed 's/ *$//')
  exp=$(printf '%s' "$expect" | tr '\n' ' ' | sed 's/ *$//')
  if [ "$got" == "$exp" ]; then
    echo "OK   $name"
  else
    echo "FAIL $name : got=[$got] expect=[$exp]"
  fi
}

# Sample from the statement: a=[1 2 1 3 1 2], queries.
# range [1,6] all: counts 1->3,2->2,3->1 => 9+4+1=14
# range [1,3] (1 2 1): 1->2,2->1 => 4+1=5
# range [2,4] (2 1 3): all distinct => 3
# range [4,4] (3): => 1
check "sample" "6 4
1 2 1 3 1 2
1 6
1 3
2 4
4 4" "14
5
3
1"

# n=1 single element, single query.
check "n1" "1 1
7
1 1" "1"

# all equal, full range: count=5 => 25
check "all-equal" "5 1
4 4 4 4 4
1 5" "25"

# all distinct, full range: 5 ones => 5
check "all-distinct" "5 1
1 2 3 4 5
1 5" "5"

# many overlapping single-point queries
check "points" "4 4
2 2 3 2
1 1
2 2
3 3
4 4" "1
1
1
1"

# n=2 power-of-two boundary, adjacent same
check "n2-same" "2 3
9 9
1 1
1 2
2 2" "1
4
1"

rm -rf "$TMP"
