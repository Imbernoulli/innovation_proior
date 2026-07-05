#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance to stdout.

Instance = a single integer n, the number of orbital-phase bins over which the
debris-sweep density profile is discretised. testId 1..10 is a difficulty ladder
(coarse -> fine phase discretisation). Deterministic: the mapping depends only on
testId, no randomness.
"""
import sys

# phase-bin counts: small-scale ladder, coarse -> fine
LADDER = [5, 6, 8, 10, 12, 14, 16, 20, 24, 30]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    if t > len(LADDER):
        t = len(LADDER)
    print(LADDER[t - 1])


if __name__ == "__main__":
    main()
