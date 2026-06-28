#!/usr/bin/env python3
"""Random + edge-case test generator for the 2xN domino-tiling problem.

Usage: gen.py <seed>

Prints a valid input file to stdout. N is kept small enough that the
independent brute oracle (O(N) big-int DP, with full enumeration for
N<=12) stays fast; p ranges over small and large primes plus a couple
of composite moduli (the problem allows any modulus 2..1e9, the
statement promises prime but the matrix method does not depend on
primality, so we stress composites too).
"""
import random
import sys

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 101]
BIG_PRIMES = [999999937, 1000000007 % (10**9 + 1), 1000000007, 999999893,
              998244353, 1000003, 1000033]
COMPOSITES = [4, 6, 8, 9, 10, 12, 100, 1000, 1000000, 999999999]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    cases = []

    # Always include a few fixed edge cases at the front for some seeds.
    if seed % 7 == 0:
        cases.append((0, rng.choice(SMALL_PRIMES)))
        cases.append((1, rng.choice(SMALL_PRIMES)))
        cases.append((2, 2))
        cases.append((0, 2))
        cases.append((1, 1000000007))

    nqueries = rng.randint(1, 8)
    for _ in range(nqueries):
        bucket = rng.random()
        if bucket < 0.5:
            n = rng.randint(0, 25)        # tiny: enumeration range overlap
        elif bucket < 0.85:
            n = rng.randint(0, 2000)      # small DP range
        else:
            n = rng.randint(0, 20000)     # larger DP range (still brute-able)

        pbucket = rng.random()
        if pbucket < 0.5:
            p = rng.choice(SMALL_PRIMES)
        elif pbucket < 0.8:
            p = rng.choice(BIG_PRIMES)
        else:
            p = rng.choice(COMPOSITES)
        cases.append((n, p))

    print(len(cases))
    for n, p in cases:
        print(n, p)


if __name__ == "__main__":
    main()
