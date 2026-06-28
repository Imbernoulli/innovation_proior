#!/usr/bin/env python3
# Random small-case generator for the mailbox placement problem.
# Usage: python3 gen.py <seed>
# Emits: "n p\n" then n sorted positions on one line.
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep cases small so the O(p n^2) brute is fast; mix sizes and value scales.
    n = rng.randint(1, 14)
    p = rng.randint(1, n)  # 1 <= p <= n

    # Value strategy: sometimes tight (forces duplicates / adjacency), sometimes wide.
    mode = rng.randint(0, 3)
    if mode == 0:
        hi = rng.randint(1, 5)          # tight range -> many duplicates
    elif mode == 1:
        hi = rng.randint(10, 50)
    elif mode == 2:
        hi = rng.randint(100, 1000)
    else:
        hi = rng.randint(1, 10 ** 6)    # wide range

    xs = sorted(rng.randint(0, hi) for _ in range(n))

    print(n, p)
    print(" ".join(map(str, xs)))


if __name__ == "__main__":
    main()
