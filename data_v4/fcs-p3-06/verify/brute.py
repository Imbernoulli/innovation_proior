#!/usr/bin/env python3
"""
Independent oracle for the derangement modulo problem in context.md.

This intentionally does not use sol.cpp's recurrence.  It computes the exact
integer count by inclusion-exclusion:

    D(n) = sum_{k=0}^n (-1)^k * C(n, k) * (n-k)!

For n <= 8 it also counts literal permutations and asserts both definitions
agree.  The oracle is for differential testing small/medium n, not for the
full 1e7 performance range.
"""

from functools import lru_cache
from itertools import permutations
from math import comb, factorial
import sys


ENUM_CAP = 8


@lru_cache(maxsize=None)
def derangement_by_inclusion(n):
    total = 0
    for k in range(n + 1):
        term = comb(n, k) * factorial(n - k)
        total += term if k % 2 == 0 else -term
    return total


@lru_cache(maxsize=None)
def derangement_by_enumeration(n):
    if n == 0:
        return 1
    count = 0
    for perm in permutations(range(n)):
        if all(perm[i] != i for i in range(n)):
            count += 1
    return count


def oracle(n, p):
    exact = derangement_by_inclusion(n)
    if n <= ENUM_CAP:
        enumerated = derangement_by_enumeration(n)
        assert exact == enumerated, (n, exact, enumerated)
    return exact % p


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    p = int(data[1])
    ns = [int(x) for x in data[2:]]
    assert len(ns) == t, (t, len(ns))
    sys.stdout.write("\n".join(str(oracle(n, p)) for n in ns))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
