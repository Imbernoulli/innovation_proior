#!/usr/bin/env python3
# Random + edge-case generator for the "min perfect squares" problem.
# Usage: gen.py <seed>
import random
import sys

EDGE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 18,
        23, 28, 31, 100, 9999, 10000, 999999, 1000000]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    # First few seeds emit deterministic edge cases for coverage.
    if seed < len(EDGE):
        print(EDGE[seed])
        return
    bucket = seed % 5
    if bucket == 0:
        n = random.randint(0, 50)          # tiny
    elif bucket == 1:
        n = random.randint(0, 2000)        # small
    elif bucket == 2:
        n = random.randint(0, 100000)      # medium
    elif bucket == 3:
        k = random.randint(1, 1000)        # perfect squares themselves
        n = k * k
    else:
        n = random.randint(0, 1000000)     # full range
    print(n)


if __name__ == "__main__":
    main()
