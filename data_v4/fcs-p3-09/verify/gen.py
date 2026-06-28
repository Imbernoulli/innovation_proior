#!/usr/bin/env python3
"""Random + edge-case test generator for the 3xN domino tiling problem.

Usage: gen.py <seed>
Prints a single test "N m" on one line.

Brute is O(N * states) so we keep N small here for the differential test; the
fast solution is exercised separately on large N for spot-checks.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    bucket = seed % 10
    if bucket == 0:
        N = 0
    elif bucket == 1:
        N = 1  # odd -> 0
    elif bucket == 2:
        N = rng.choice([2, 3, 4, 5])
    elif bucket == 3:
        N = rng.randint(0, 60)
    else:
        N = rng.randint(0, 300)

    mbucket = seed % 7
    if mbucket == 0:
        m = 1
    elif mbucket == 1:
        m = 2
    elif mbucket == 2:
        m = rng.choice([1000000007, 998244353])
    elif mbucket == 3:
        m = rng.choice([10, 100, 1000])  # composite moduli
    else:
        m = rng.randint(1, 10 ** 9)

    print(N, m)


if __name__ == "__main__":
    main()
