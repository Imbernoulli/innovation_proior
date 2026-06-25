import sys
from fractions import Fraction

# Exact DP oracle (same problem) using Python big ints / Fraction.
# Works for large n (no subset enumeration), used to validate adversarial
# overflow cases that brute.py cannot reach.

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    cal = []; mass = []
    for _ in range(n):
        cal.append(int(data[idx])); idx += 1
        mass.append(int(data[idx])); idx += 1
    sumMass = sum(mass)
    NEG = None
    dp = [NEG] * (sumMass + 1)
    dp[0] = 0
    for i in range(n):
        m = mass[i]; c = cal[i]
        for w in range(sumMass, m - 1, -1):
            if dp[w - m] is not None:
                cand = dp[w - m] + c
                if dp[w] is None or cand > dp[w]:
                    dp[w] = cand
    best = None
    for W in range(L, sumMass + 1):
        if dp[W] is None:
            continue
        frac = Fraction(dp[W], W)
        if best is None or frac > best:
            best = frac
    if best is None:
        print("IMPOSSIBLE")
    else:
        print(f"{best.numerator}/{best.denominator}")

main()
