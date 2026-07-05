#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance (the network size N) to stdout.

The difficulty ladder is the order N: odd primes from 7 up to 41. Odd/prime N is chosen so no
Hadamard matrix exists and the maximal +/-1 determinant is genuinely open (no known optimum).
The instance is fully specified by N; the solver constructs the N x N +/-1 coupling matrix.
"""
import sys

# odd primes only -> not a multiple of 4, no Hadamard order, no closed-form optimum
PRIMES = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(2)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > len(PRIMES):
        t = len(PRIMES)
    N = PRIMES[t - 1]
    sys.stdout.write("%d\n" % N)


if __name__ == "__main__":
    main()
