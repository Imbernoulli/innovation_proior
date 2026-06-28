#!/usr/bin/env python3
"""
Random small-case generator for "Aliens-Trick Job Split".

Usage: python3 gen.py <seed>

Emits:
    n k
    a_0 ... a_{n-1}

Feasibility: segments are non-empty and separated by >=1 unused cell, so a
valid choice of exactly k segments exists iff 1 <= k <= ceil(n/2). We always
pick k in [1, ceil(n/2)].
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 12)
    kmax = (n + 1) // 2          # ceil(n/2): max gap-separated non-empty segments
    k = rng.randint(1, kmax)

    mode = rng.randint(0, 5)
    if mode == 0:
        vals = [rng.randint(-9, 9) for _ in range(n)]
    elif mode == 1:
        vals = [rng.randint(1, 9) for _ in range(n)]        # all positive
    elif mode == 2:
        vals = [rng.randint(-9, -1) for _ in range(n)]      # all negative
    elif mode == 3:
        vals = [rng.choice([-5, 0, 5]) for _ in range(n)]   # zeros + extremes
    elif mode == 4:
        vals = [rng.randint(-3, 3) for _ in range(n)]       # small, many ties
    else:
        vals = [rng.choice([-1000, -1, 0, 1, 1000]) for _ in range(n)]  # wide

    out = [f"{n} {k}", " ".join(map(str, vals))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
