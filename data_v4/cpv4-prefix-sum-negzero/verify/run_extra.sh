#!/usr/bin/env bash
SP="$1"
V=/srv/home/bohanlyu/innovation_proior/data_v4/cpv4-prefix-sum-negzero/verify
python3 -c "print(200000); print(' '.join(['-1000000000']*200000))" > "$SP/big.txt"
echo -n "overflow result: "
/tmp/cpv4-prefix-sum-negzero_sol < "$SP/big.txt"
echo "expected 200000000000000"
pass=0; mismatch=0
for s in $(seq 401 700); do
  python3 "$V/gen.py" "$s" > "$SP/in.txt"
  a=$(/tmp/cpv4-prefix-sum-negzero_sol < "$SP/in.txt")
  b=$(python3 "$V/brute.py" < "$SP/in.txt")
  if [ "$a" == "$b" ]; then pass=$((pass+1)); else mismatch=$((mismatch+1)); echo "FAIL seed=$s"; cat "$SP/in.txt"; fi
done
echo "EXTRA PASS=$pass MISMATCH=$mismatch"
rm -f "$SP/big.txt" "$SP/in.txt"
