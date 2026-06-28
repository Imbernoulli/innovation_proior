#!/usr/bin/env python3
"""Instance generator for the Number-Link (Flow-Free style) maximum-connections
problem.

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the exact stdin format consumed by sol.cpp:

    H W K
    r1a c1a r1b c1b      # endpoints of pair 0 (color 0)
    r2a c2a r2b c2b      # endpoints of pair 1 (color 1)
    ...
    rKa cKa rKb cKb      # endpoints of pair K-1 (color K-1)

All coordinates are 0-indexed: 0 <= r < H, 0 <= c < W.  Every endpoint cell is
distinct (the 2*K endpoint cells are pairwise different).  Grids are roughly
square and "full-ish" so that connecting every pair simultaneously is generally
impossible -- that is what makes the maximisation NP-hard and interesting.
"""
import sys
import random


def gen(seed: int):
    rng = random.Random(seed * 2654435761 + 12345)

    # Grid size: between 12 and 30 on a side, biased toward larger / square.
    H = rng.randint(12, 30)
    W = rng.randint(12, 30)
    n = H * W

    # Number of pairs.  We keep the instance in an INTERESTING density band:
    # endpoints occupy roughly 25%-40% of the cells (pairs ~ 0.125*n to 0.20*n).
    # Sparser than this and nearly everything routes (no discrimination); denser
    # and the grid saturates with endpoint obstacles so almost nothing routes.
    # In this band many -- but not all -- pairs can be routed simultaneously, so
    # the maximisation is genuinely NP-hard and the heuristic has real headroom.
    lo_pairs = max(2, (n * 25) // 200)   # ~0.125 * n  (25% endpoint occupancy)
    hi_pairs = max(lo_pairs + 1, (n * 40) // 200)  # ~0.20 * n (40% occupancy)
    K = rng.randint(lo_pairs, hi_pairs)

    cells = [(r, c) for r in range(H) for c in range(W)]
    rng.shuffle(cells)

    # Take 2*K distinct cells as endpoints.
    chosen = cells[: 2 * K]

    pairs = []
    for i in range(K):
        a = chosen[2 * i]
        b = chosen[2 * i + 1]
        pairs.append((a, b))

    out = []
    out.append(f"{H} {W} {K}")
    for (a, b) in pairs:
        out.append(f"{a[0]} {a[1]} {b[0]} {b[1]}")
    sys.stdout.write("\n".join(out) + "\n")


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    gen(seed)


if __name__ == "__main__":
    main()
