#!/usr/bin/env python3
"""Instance generator for ale-29 "Connected Region Selection".

Usage: python3 gen.py SEED  > instance.txt

Output format (stdin contract of the solver):
    line 1: H W B
    next H lines: W integers each, the weight grid w[r][c]  (may be negative)

The instances are deliberately "patchy": a smooth low-frequency field (a sum of
random 2D Gaussian bumps, some positive, some negative) plus per-cell noise, then
shifted so the field has both clearly profitable and clearly toxic regions. This
makes the optimal connected region a non-convex blob with notches -- exactly the
structure where a connectivity-aware local search beats a greedy growth.
"""
import sys
import math
import random


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py SEED\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(1_000_003 * seed + 12345)

    # Grid size and budget. Kept fixed across seeds so scores are comparable.
    H = 60
    W = 60
    # Budget: a region of up to B cells (about a quarter of the grid).
    B = 900

    # Build a smooth field from K Gaussian bumps with mixed signs.
    K = rng.randint(10, 16)
    bumps = []
    for _ in range(K):
        cy = rng.uniform(0, H - 1)
        cx = rng.uniform(0, W - 1)
        amp = rng.uniform(4.0, 14.0) * (1 if rng.random() < 0.55 else -1)
        sig = rng.uniform(4.0, 11.0)
        bumps.append((cy, cx, amp, sig))

    grid = [[0 for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            v = 0.0
            for (cy, cx, amp, sig) in bumps:
                d2 = (r - cy) ** 2 + (c - cx) ** 2
                v += amp * math.exp(-d2 / (2.0 * sig * sig))
            # per-cell noise
            v += rng.gauss(0.0, 2.0)
            # global downward shift so plenty of cells are negative (toxic)
            v -= 1.5
            grid[r][c] = int(round(v))

    out = []
    out.append(f"{H} {W} {B}")
    for r in range(H):
        out.append(" ".join(str(grid[r][c]) for c in range(W)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
