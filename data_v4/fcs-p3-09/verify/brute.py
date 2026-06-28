#!/usr/bin/env python3
"""Independent brute oracle for tilings of a 3xN board mod m.

Method: straightforward column-by-column profile DP that *enumerates* every
domino placement directly (no matrix exponentiation, no closed-form recurrence,
no precomputed table). Runs in O(N * states) so it is only used for small N to
cross-check the fast solution. Computes the count modulo m as it goes.
"""
import sys
from collections import defaultdict


def column_fillings(cur_mask, R=3):
    """Yield protrusion masks for the next column, one per complete filling of
    this column given that cells in cur_mask are already filled."""
    results = []

    def rec(r, filled, nxt):
        if r == R:
            results.append(nxt)
            return
        if filled & (1 << r):
            rec(r + 1, filled, nxt)
            return
        # vertical domino over rows r, r+1
        if r + 1 < R and not (filled & (1 << (r + 1))):
            rec(r + 2, filled | (1 << r) | (1 << (r + 1)), nxt)
        # horizontal domino protruding into next column from row r
        rec(r + 1, filled | (1 << r), nxt | (1 << r))

    rec(0, cur_mask, 0)
    return results


def tilings(N, m):
    if m == 1:
        return 0
    # dp over columns: dp[mask] = ways to reach this column with given prefilled mask
    dp = {0: 1 % m}
    for _ in range(N):
        nxt = defaultdict(int)
        for mask, cnt in dp.items():
            for nm in column_fillings(mask):
                nxt[nm] = (nxt[nm] + cnt) % m
        dp = nxt
    return dp.get(0, 0) % m


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    m = int(data[1])
    print(tilings(N, m))


if __name__ == "__main__":
    main()
