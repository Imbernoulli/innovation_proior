#!/bin/bash
D=/srv/home/bohanlyu/innovation_proior/data_v4/fcs-p2-05/verify
SOL=$D/sol
python3 -c "
import random
r=random.Random(7)
n=18
print(n)
for i in range(n):
    print(' '.join(str(r.randint(0,10**9)) for _ in range(n)))
" > /tmp/n18.txt
start=$(date +%s.%N)
out=$($SOL < /tmp/n18.txt)
end=$(date +%s.%N)
echo "n=18 sol output: $out"
echo "elapsed seconds: $(python3 -c "print(f'{$end-$start:.3f}')")"
rm -f /tmp/n18.txt
