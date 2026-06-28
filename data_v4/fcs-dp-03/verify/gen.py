#!/usr/bin/env python3
"""Random small-case generator for the optimal adjacent-merge problem.

Usage: python3 gen.py <seed>

Emits a case on stdout: first line n, second line n space-separated weights.
Sizes are kept small so the O(n^3) oracle stays fast. We mix in a few regimes:
tiny n (including 0 and 1), small weights (to force many ties in the DP), and
occasional larger weights (to exercise 64-bit accumulation paths).
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Choose n. Bias toward small but include the degenerate 0 and 1 sometimes.
    roll = rng.random()
    if roll < 0.05:
        n = 0
    elif roll < 0.12:
        n = 1
    elif roll < 0.20:
        n = 2
    else:
        n = rng.randint(2, 40)

    # Weight regime. Weights are positive (piles of stones), occasionally large.
    regime = rng.random()
    if regime < 0.5:
        hi = rng.randint(1, 9)          # many ties
    elif regime < 0.8:
        hi = rng.randint(10, 1000)
    else:
        hi = rng.randint(10 ** 6, 10 ** 9)

    w = [rng.randint(1, hi) for _ in range(n)]

    out = [str(n)]
    if n > 0:
        out.append(" ".join(map(str, w)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
