#!/usr/bin/env bash
set -u
BIN="/tmp/fcs-gx-02_x"
BRUTE="python3 /srv/home/bohanlyu/innovation_proior/data_v4/fcs-gx-02/verify/brute.py"
run() {
  local name="$1" inp="$2"
  local o1 o2
  o1=$(printf "%s" "$inp" | $BIN)
  o2=$(printf "%s" "$inp" | $BRUTE)
  if [ "$o1" == "$o2" ]; then
    echo "OK   $name : sol=[$o1]"
  else
    echo "FAIL $name : sol=[$o1] brute=[$o2]"
  fi
}
run "remove-k-digits sample" $'1432219\n3\n'
run "all-decreasing"         $'edcba\n2\n'
run "all-increasing"        $'abcde\n2\n'
run "all-same"              $'aaaaa\n2\n'
run "k=0 no deletion"       $'bca\n0\n'
run "k=n delete all"        $'abc\n3\n'
run "k>n delete more"       $'ab\n5\n'
run "single char keep"      $'z\n0\n'
run "single char delete"    $'z\n1\n'
run "leading-zero digits"   $'10200\n1\n'
run "leading-zero to empty" $'10\n2\n'
run "zigzag"                $'bacacb\n3\n'
run "tie then small"        $'112112\n3\n'
run "classic 100 k1"        $'100\n1\n'
run "all nines"             $'9999\n2\n'
