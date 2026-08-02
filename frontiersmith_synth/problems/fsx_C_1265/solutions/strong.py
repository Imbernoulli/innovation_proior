# TIER: strong
"""
The genuine insight has two parts, and both are needed:

1. RECOMPUTE THE RIGHT QUANTITY (vertical-heat-path + hotspot-coupling): heat generated on
   die m must cross m layer-boundaries of resistance to reach the sink, and every die sharing
   a column shares that same vertical path -- so what determines a column's peak temperature
   is the depth-weighted, STACKED profile
       W[c] = sum_{m=1}^{M} m * p[m][c],
   never any single die's own power reading. This alone already recovers hotspots that a
   per-die view cannot see (lower dies contribute at lower weight each, but there can be many
   of them, and they all funnel down the same via).

2. RECOGNIZE THE OBJECTIVE'S SHAPE (tsv-area-overhead under a min-max, not a sum): the score is
   driven by max_c R(c)*W[c] -- the single worst remaining column -- not by a total. Whichever
   column currently has the largest W[c] and is NOT via'd sets the score; nothing is gained by
   optimizing a value-per-cost ratio the way you would for a sum-maximizing knapsack. The
   right move is an exchange argument: process columns in decreasing W[c] order and via every
   one you can still afford. Any column skipped for lack of budget remains the binding column
   regardless of what you do with cheaper, lower-W columns further down the list, so there is
   never a reason to prefer a cheaper-but-lower-W column over a costlier-but-higher-W one.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    M = int(nx())
    N = int(nx())
    A = int(nx())
    nx(); nx()  # R0, Rv unused by the selection itself (Rv < R0 makes any via a strict win)
    a = [int(nx()) for _ in range(N)]
    P = [[int(nx()) for _ in range(N)] for _ in range(M)]

    W = [0] * N
    for c in range(N):
        s = 0
        for m in range(M):
            s += (m + 1) * P[m][c]
        W[c] = s

    order = sorted(range(N), key=lambda c: (-W[c], c))

    x = [0] * N
    remaining = A
    for c in order:
        if a[c] <= remaining:
            x[c] = 1
            remaining -= a[c]

    sys.stdout.write(" ".join(map(str, x)) + "\n")


if __name__ == "__main__":
    main()
