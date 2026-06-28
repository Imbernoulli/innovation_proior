#!/usr/bin/env python3
"""Random + edge-case test generator for fcs-p3-04.

Usage: gen.py <seed> [maxn]

Emits a single test file on stdout in the problem's input format:
    T
    N_1 p_1
    N_2 p_2
    ...

By default N is kept small/medium (<= maxn, default 4000) so the big-integer
brute oracle in brute.py stays fast; large-N coverage is handled separately by
the harness (test.py) which checks sol.cpp against an independent Python
matrix-power reference at N up to 1e18.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    rng = random.Random(seed)

    cases = []

    # A handful of deterministic edge cases sprinkled in.
    edges = [
        (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 7),
        (0, 1000000000000000000), (1, 1000000000000000000),
        (5, 1), (10, 13), (22, 999983), (23, 2),
    ]
    for (n, p) in edges:
        if rng.random() < 0.5:
            cases.append((n, p))

    ncases = rng.randint(1, 12)
    for _ in range(ncases):
        r = rng.random()
        if r < 0.30:
            n = rng.randint(0, 22)            # enumerable range
        elif r < 0.70:
            n = rng.randint(0, 300)           # small DP range
        else:
            n = rng.randint(0, maxn)          # medium DP range

        pr = rng.random()
        if pr < 0.20:
            p = rng.randint(1, 20)            # tiny moduli, incl. 1 and 2
        elif pr < 0.55:
            p = rng.choice([2, 3, 5, 7, 11, 13, 97, 998244353, 1000000007])
        elif pr < 0.80:
            p = rng.randint(1, 10**9)
        else:
            p = rng.randint(1, 10**18)        # full-range modulus
        cases.append((n, p))

    rng.shuffle(cases)
    out = [str(len(cases))]
    for (n, p) in cases:
        out.append(f"{n} {p}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
