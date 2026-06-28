#!/usr/bin/env python3
"""Random test generator for Burst Balloons.

Usage: gen.py SEED [MAXN] [MAXV]
Emits stdin in the judged format: first line n, second line the n values.
n is kept small (<= ~9) so the independent brute oracle stays fast; values
are pushed to include 0s and the max value to exercise the value range.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    maxv = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    rng = random.Random(seed)

    n = rng.randint(0, maxn)
    vals = []
    for _ in range(n):
        r = rng.random()
        if r < 0.15:
            vals.append(0)              # exercise zero balloons
        elif r < 0.30:
            vals.append(maxv)           # exercise the cap
        else:
            vals.append(rng.randint(0, maxv))

    out = [str(n)]
    out.append(" ".join(map(str, vals)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
