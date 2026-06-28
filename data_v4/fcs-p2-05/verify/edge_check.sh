#!/bin/bash
set -e
D=/srv/home/bohanlyu/innovation_proior/data_v4/fcs-p2-05/verify
SOL=$D/sol
echo "n=0:"; printf "0\n" | $SOL
echo "n=1 negative (-5):"; printf "1\n-5\n" | $SOL
python3 -c "
import random
r=random.Random(12345)
n=9
print(n)
for i in range(n):
    print(' '.join(str(r.randint(0,1000)) for _ in range(n)))
" > /tmp/n9.txt
echo "n9 sol:"; $SOL < /tmp/n9.txt
echo "n9 brute:"; python3 $D/brute.py < /tmp/n9.txt
rm -f /tmp/n9.txt
