#!/usr/bin/env python3
"""Independent brute-force oracle for palindrome partitioning minimum cuts.

Strategy: exhaustive recursion with memoization over the start index.
best(i) = minimum cuts needed to partition s[i:] into palindromes,
counting the number of CUTS (= number of pieces - 1).
For every prefix s[i:j+1] that is a palindrome, recurse on the remainder.
This is a deliberately different formulation from the DP in sol.cpp
(recursive front-to-back with explicit palindrome check via slicing/reversal),
so it serves as an independent check.
"""
import sys
from functools import lru_cache


def solve(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0

    def is_pal(a: int, b: int) -> bool:  # s[a:b] inclusive-exclusive
        sub = s[a:b]
        return sub == sub[::-1]

    sys.setrecursionlimit(10000)

    @lru_cache(maxsize=None)
    def best(i: int) -> int:
        # minimum number of pieces to cover s[i:]
        if i == n:
            return 0
        res = None
        for j in range(i + 1, n + 1):
            if is_pal(i, j):
                cand = 1 + best(j)
                if res is None or cand < res:
                    res = cand
        return res

    pieces = best(0)
    return pieces - 1  # cuts = pieces - 1


def main():
    data = sys.stdin.read().split()
    s = data[0] if data else ""
    print(solve(s))


if __name__ == "__main__":
    main()
