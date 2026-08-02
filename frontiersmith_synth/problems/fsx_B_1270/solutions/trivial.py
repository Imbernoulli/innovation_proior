# TIER: trivial
#!/usr/bin/env python3
"""Spread the whole premium budget evenly (same % participation share) over
every layer in the catalog -- exactly the checker's own reference construction,
so this scores ~0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    catalog = []
    for _ in range(M):
        A = int(next(it)); W = int(next(it)); Prem = int(next(it))
        K = int(next(it)); RP = int(next(it))
        catalog.append((A, W, Prem, K, RP))
    C0 = int(next(it))
    Pmax = int(next(it))
    S = int(next(it))
    for _ in range(S):
        n = int(next(it))
        for _ in range(n):
            next(it)

    total_prem = sum(c[2] for c in catalog)
    p = min(100, (100 * Pmax) // total_prem) if total_prem > 0 else 0
    shares = [int(p)] * M
    print(" ".join(map(str, shares)))


if __name__ == "__main__":
    main()
