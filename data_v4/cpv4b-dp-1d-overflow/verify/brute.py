import sys
from itertools import product

# Brute force: enumerate every length-n sequence of scores in 0..m,
# count those with no two consecutive peak scores (peak == m).
# Score "is peak" iff value == m. Count mod 1e9+7.
# Only used for small n, m.

def main():
    data = sys.stdin.read().split()
    n = int(data[0]); m = int(data[1])
    MOD = 1000000007
    if n == 0:
        print(1 % MOD)
        return
    # values 0..m inclusive  => m+1 possible scores
    count = 0
    for seq in product(range(m + 1), repeat=n):
        ok = True
        for j in range(1, n):
            if seq[j] == m and seq[j - 1] == m:
                ok = False
                break
        if ok:
            count += 1
    print(count % MOD)

main()
