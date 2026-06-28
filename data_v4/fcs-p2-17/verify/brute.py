#!/usr/bin/env python3
"""Independent brute oracle for the coin-multiset counting problem.

Problem: count the number of distinct multisets of coins (unlimited supply,
order does NOT matter) summing to exactly S, modulo a prime p. A "way" is fixed
by how many coins of each DISTINCT denomination value are used, so duplicate
denomination values in the input are collapsed.

Independent formulation: explicit recursive enumeration over the distinct
denominations. We process denominations one at a time and, for the current one,
sum over every possible *count* k = 0, 1, 2, ... of that coin (k * value <=
remaining), recursing on the rest. This counts each multiset exactly once by
construction (we fix a per-denomination usage count), with no notion of order at
all -- a fundamentally different structure from the forward sum-relaxation DP in
sol.cpp, so it is a real cross-check.

To stay fast enough for the small differential-test sizes, we memoize on
(denomination index, remaining sum). The arithmetic is done in plain Python big
integers and reduced mod p only at the end, so the oracle's correctness does not
depend on any modular-reduction reasoning of its own.

Reads the same stdin format as sol.cpp:
    n S p
    c[0] c[1] ... c[n-1]
Prints the number of distinct multisets mod p.
"""
import sys
from functools import lru_cache


def solve(data):
    it = iter(data.split())
    try:
        n = int(next(it))
        S = int(next(it))
        p = int(next(it))
    except StopIteration:
        return None
    coins = [int(next(it)) for _ in range(n)]

    # Collapse duplicate denomination values; keep only those that can ever be
    # used (value <= S). Sorting is only for determinism.
    vals = sorted(set(v for v in coins if 0 < v <= S))
    m = len(vals)

    sys.setrecursionlimit(1000000)

    @lru_cache(maxsize=None)
    def count(idx, rem):
        # Number of multisets using only denominations vals[idx:] that sum to rem.
        if rem == 0:
            return 1
        if idx == m:
            return 0
        v = vals[idx]
        total = 0
        k = 0
        # Use exactly k copies of vals[idx], then move on to the next denomination.
        while k * v <= rem:
            total += count(idx + 1, rem - k * v)
            k += 1
        return total

    ans = count(0, S)
    return ans % p


def main():
    data = sys.stdin.read()
    res = solve(data)
    if res is None:
        return
    print(res)


if __name__ == "__main__":
    main()
