#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance of the greenhouse-pod dispersion problem.

testId 1..10 is a difficulty ladder: the number of monitoring pods `n` grows.
All values are a deterministic function of testId only (bit-for-bit reproducible);
there is no randomness in the instance itself.

Instance format (stdin of the solver):
    n

`n` is always chosen to be PRIME.  Primality is what makes the checker's internal
baseline construction (a quadratic-residue point set) guaranteed to have every
triangle non-degenerate, so the baseline is always strictly positive and feasible.
The number of triples grows as C(n,3), so larger testId = a genuinely larger,
harder max-min-area layout problem.
"""
import sys

# 10 increasing primes -> the difficulty ladder (small .. large).
PRIMES = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    if tid > len(PRIMES):
        tid = len(PRIMES)
    n = PRIMES[tid - 1]
    sys.stdout.write("%d\n" % n)


if __name__ == "__main__":
    main()
