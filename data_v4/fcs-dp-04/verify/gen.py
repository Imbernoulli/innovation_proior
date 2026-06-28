#!/usr/bin/env python3
"""Random small-case generator for the domino-tiling-count problem.

Usage: gen.py <seed>
Emits a valid instance on stdout:
    h w p
    h lines of '.'/'#'

Kept small enough for the backtracking oracle: h,w in [1..5], so total cells <= 25.
'#' density varies so we hit dense, sparse, all-free and heavily-blocked boards.
The modulus p is chosen from a small set of primes (including 2 and a big one).
"""
import sys
import random

PRIMES = [2, 3, 5, 7, 998244353, 1000000007, 1000000009]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    h = rng.randint(1, 5)
    w = rng.randint(1, 5)
    p = rng.choice(PRIMES)

    # density of '#'
    block_prob = rng.choice([0.0, 0.1, 0.2, 0.3, 0.5, 0.7])

    grid = []
    for _ in range(h):
        row = []
        for _ in range(w):
            row.append('#' if rng.random() < block_prob else '.')
        grid.append(''.join(row))

    out = [f"{h} {w} {p}"]
    out.extend(grid)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
