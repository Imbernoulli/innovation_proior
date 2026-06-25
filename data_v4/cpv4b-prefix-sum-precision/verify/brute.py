import sys
from fractions import Fraction
from math import gcd

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n

    # prefix sums
    S = [0] * (n + 1)
    for i in range(n):
        S[i + 1] = S[i] + a[i]

    best = None  # (num, den) representing maximum average, den > 0
    # iterate all windows [i, j) with length >= L  (i < j, j-i >= L)
    for i in range(0, n + 1):
        for j in range(i + L, n + 1):
            num = S[j] - S[i]
            den = j - i
            if best is None:
                best = (num, den)
            else:
                # compare num/den vs best
                if num * best[1] > best[0] * den:
                    best = (num, den)

    num, den = best
    g = gcd(abs(num), den)
    if g == 0:
        g = 1
    num //= g
    den //= g
    print(f"{num}/{den}")

main()
