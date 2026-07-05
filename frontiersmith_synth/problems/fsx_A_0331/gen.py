#!/usr/bin/env python3
"""gen.py <testId> -> one 'orbital debris cleanup' instance on stdout.

Difficulty ladder over testId 1..10: signature length n grows from 4 to 7. A
deterministic (seeded by testId) set of 'protected' signatures is carved out so each
instance is a DIFFERENT cap-set-avoiding-a-forbidden-set problem whose optimum is not
tabulated. Only testId seeds the randomness -> fully reproducible.
"""
import sys, random

NS = [4, 4, 5, 5, 6, 6, 6, 7, 7, 7]
FORBID_FRAC = 0.15


def digits(v, n):
    d = []
    for _ in range(n):
        d.append(v % 3)
        v //= 3
    return d


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    if tid > len(NS):
        tid = len(NS)
    n = NS[tid - 1]
    N = 3 ** n
    rng = random.Random(1000 * tid + n)
    m = int(FORBID_FRAC * N)
    forb = rng.sample(range(N), m)
    out = ["%d %d" % (n, m)]
    for v in forb:
        out.append(" ".join(str(x) for x in digits(v, n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
