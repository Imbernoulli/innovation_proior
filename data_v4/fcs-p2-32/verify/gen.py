#!/usr/bin/env python3
"""Random test generator for the max-dot-product problem.

Usage: gen.py SEED [MAXN MAXV]
Keeps n, m small enough that the brute oracle is feasible, and biases toward
negatives / mixed signs (the regime where greedy pairing breaks).
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    maxv = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    rng = random.Random(seed)

    n = rng.randint(1, maxn)
    m = rng.randint(1, maxn)

    mode = rng.randint(0, 3)
    if mode == 0:
        lo, hi = -maxv, maxv          # mixed signs (default)
    elif mode == 1:
        lo, hi = -maxv, -1            # all negative
    elif mode == 2:
        lo, hi = 1, maxv              # all positive
    else:
        lo, hi = -maxv, maxv
        # occasionally inject zeros
    A = [rng.randint(lo, hi) for _ in range(n)]
    B = [rng.randint(lo, hi) for _ in range(m)]
    if mode == 3:
        for arr in (A, B):
            for k in range(len(arr)):
                if rng.random() < 0.3:
                    arr[k] = 0

    out = [f"{n} {m}"]
    out.append(" ".join(map(str, A)))
    out.append(" ".join(map(str, B)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
