#!/usr/bin/env python3
"""
Random test generator for the "max subarray with at most one deletion" problem.

Usage: gen.py <seed> [mode]
Prints a test case to stdout in the format:
  n
  a[0] a[1] ... a[n-1]

Modes bias toward edge-prone distributions: all-negative, mixed small range,
single element, alternating signs, large magnitudes, etc.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    rng = random.Random(seed)

    if mode is None:
        mode = rng.choice([
            "tiny", "small_range", "all_neg", "all_pos",
            "alternating", "single", "big_mag", "mixed", "zeros",
        ])

    if mode == "tiny":
        n = rng.randint(1, 6)
        a = [rng.randint(-5, 5) for _ in range(n)]
    elif mode == "small_range":
        n = rng.randint(1, 12)
        a = [rng.randint(-3, 3) for _ in range(n)]
    elif mode == "all_neg":
        n = rng.randint(1, 12)
        a = [rng.randint(-20, -1) for _ in range(n)]
    elif mode == "all_pos":
        n = rng.randint(1, 12)
        a = [rng.randint(1, 20) for _ in range(n)]
    elif mode == "alternating":
        n = rng.randint(1, 14)
        a = []
        for i in range(n):
            v = rng.randint(1, 15)
            a.append(v if i % 2 == 0 else -v)
    elif mode == "single":
        n = 1
        a = [rng.randint(-1000, 1000)]
    elif mode == "big_mag":
        n = rng.randint(1, 10)
        a = [rng.randint(-10**9, 10**9) for _ in range(n)]
    elif mode == "zeros":
        n = rng.randint(1, 12)
        a = [rng.choice([0, 0, 0, rng.randint(-5, 5)]) for _ in range(n)]
    else:  # mixed
        n = rng.randint(1, 16)
        a = [rng.randint(-12, 12) for _ in range(n)]

    print(n)
    print(" ".join(map(str, a)))


if __name__ == "__main__":
    main()
