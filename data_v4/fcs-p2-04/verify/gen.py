#!/usr/bin/env python3
"""Random + edge-case generator for the equal-sum partition problem.

Usage: gen.py SEED [MODE]
Prints a valid stdin instance: n, then n integers (1..1000), n <= 200.

MODE controls the distribution so we exercise both YES- and NO-heavy
regimes, parity corners, and the n / value extremes.
"""
import random
import sys

VMAX = 1000
NMAX = 200


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    rng = random.Random(seed)

    if mode is None:
        mode = rng.choice([
            "small", "small", "small",        # tiny n -> exhaustive oracle
            "tiny_vals", "mid", "yes_planted",
            "odd_total", "all_same", "extreme_n", "extreme_vals",
            "one", "two", "powers",
        ])

    if mode == "small":
        n = rng.randint(0, 16)
        a = [rng.randint(1, 12) for _ in range(n)]
    elif mode == "tiny_vals":
        n = rng.randint(0, 18)
        a = [rng.randint(1, 3) for _ in range(n)]
    elif mode == "mid":
        n = rng.randint(0, 18)
        a = [rng.randint(1, VMAX) for _ in range(n)]
    elif mode == "yes_planted":
        # build a multiset that definitely has an equal split, then maybe shuffle
        k = rng.randint(1, 8)
        base = [rng.randint(1, VMAX) for _ in range(k)]
        a = base + base[:]               # duplicate so a perfect split exists
        rng.shuffle(a)
    elif mode == "odd_total":
        n = rng.randint(1, 18)
        a = [rng.randint(1, VMAX) for _ in range(n)]
        # force odd total
        if sum(a) % 2 == 0:
            a[0] += 1
    elif mode == "all_same":
        n = rng.randint(0, 18)
        v = rng.randint(1, VMAX)
        a = [v] * n
    elif mode == "extreme_n":
        n = NMAX
        a = [rng.randint(1, VMAX) for _ in range(n)]
    elif mode == "extreme_vals":
        n = rng.randint(0, 18)
        a = [rng.choice([1, VMAX]) for _ in range(n)]
    elif mode == "one":
        a = [rng.randint(1, VMAX)]
    elif mode == "two":
        a = [rng.randint(1, VMAX), rng.randint(1, VMAX)]
    elif mode == "powers":
        n = rng.randint(0, 16)
        a = [1 << rng.randint(0, 9) for _ in range(n)]
    else:
        n = rng.randint(0, 18)
        a = [rng.randint(1, VMAX) for _ in range(n)]

    print(len(a))
    print(" ".join(map(str, a)))


if __name__ == "__main__":
    main()
