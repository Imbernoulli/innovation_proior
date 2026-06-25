import sys
from fractions import Fraction

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    p = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    g = []
    for _ in range(n):
        g.append(int(data[idx])); idx += 1

    # Independent O(n^2) reference: for every unordered pair {i, j}, let lo = min, hi = max.
    # The pair is balanced iff hi/lo <= p/q. Use exact rational arithmetic (Fraction) so there is
    # no rounding whatsoever -- a deliberately different mechanism from the two-pointer solution.
    thresh = Fraction(p, q)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            lo = min(g[i], g[j])
            hi = max(g[i], g[j])
            if Fraction(hi, lo) <= thresh:
                count += 1
    print(count)

main()
