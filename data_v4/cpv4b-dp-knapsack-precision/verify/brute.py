import sys
from fractions import Fraction
from itertools import combinations

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    cal = []
    mass = []
    for _ in range(n):
        c = int(data[idx]); idx += 1
        m = int(data[idx]); idx += 1
        cal.append(c)
        mass.append(m)

    best = None  # Fraction
    # enumerate all non-empty subsets
    for r in range(1, n + 1):
        for comb in combinations(range(n), r):
            tw = sum(mass[i] for i in comb)
            if tw < L:
                continue
            tv = sum(cal[i] for i in comb)
            frac = Fraction(tv, tw)
            if best is None or frac > best:
                best = frac

    if best is None:
        print("IMPOSSIBLE")
    else:
        print(f"{best.numerator}/{best.denominator}")

main()
