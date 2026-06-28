#!/usr/bin/env python3
"""Random test generator for the circular max non-adjacent sum problem.

Usage: gen.py <seed>
Emits to stdout: n on the first line, then n integers.
Keeps n small (<= ~16) so the 2^n brute oracle stays fast, and biases value
ranges/edge cases to stress the wrap edge and negatives.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = seed % 7
    if mode == 0:
        n = rng.randint(0, 2)            # tiny: empty / single / pair
    elif mode == 1:
        n = 3                            # smallest "real" cycle
    elif mode == 2:
        n = rng.randint(4, 8)
    else:
        n = rng.randint(0, 16)

    # value range varies to hit all-positive, all-negative, mixed, and large
    vmode = (seed // 7) % 5
    if vmode == 0:
        lo, hi = -5, 5                   # small mixed, many ties
    elif vmode == 1:
        lo, hi = 1, 10                   # all positive
    elif vmode == 2:
        lo, hi = -10, -1                 # all negative -> answer 0
    elif vmode == 3:
        lo, hi = -1000000000, 1000000000 # extreme magnitudes (overflow check)
    else:
        lo, hi = -3, 3                   # very tight, lots of zeros/negatives

    a = [rng.randint(lo, hi) for _ in range(n)]
    print(n)
    print(" ".join(map(str, a)))


if __name__ == "__main__":
    main()
