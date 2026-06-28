#!/usr/bin/env python3
"""Independent brute oracle for Longest Common Subsequence length.

Reads two whitespace-separated tokens (strings) from stdin, prints the length
of their longest common subsequence. Uses a completely independent method from
verify/sol.cpp: for short inputs it does an exhaustive recursive search over
which characters of s are kept (memoized on (i, j)), which is a different
formulation than the rolling-row tabulation, so a transcription bug in one is
unlikely to be mirrored in the other. Falls back to the standard full-table DP
for larger inputs so it can also serve as a reference on big cases.
"""
import sys
from functools import lru_cache


def lcs_exhaustive(s, t):
    # Pure recursive top-down: at each (i, j) either characters match and we
    # consume both, or we branch by skipping one side. Memoized.
    sys.setrecursionlimit(1 << 25)

    @lru_cache(maxsize=None)
    def rec(i, j):
        if i == len(s) or j == len(t):
            return 0
        if s[i] == t[j]:
            return 1 + rec(i + 1, j + 1)
        return max(rec(i + 1, j), rec(i, j + 1))

    res = rec(0, 0)
    rec.cache_clear()
    return res


def lcs_table(s, t):
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        si = s[i - 1]
        di = dp[i]
        dim1 = dp[i - 1]
        for j in range(1, m + 1):
            if si == t[j - 1]:
                di[j] = dim1[j - 1] + 1
            else:
                di[j] = dim1[j] if dim1[j] >= di[j - 1] else di[j - 1]
    return dp[n][m]


def main():
    data = sys.stdin.read().split()
    if len(data) == 0:
        print(0)
        return
    if len(data) == 1:
        # only one token present -> the other string is empty
        print(0)
        return
    s, t = data[0], data[1]
    # Use exhaustive memoized search when small (true independence); otherwise
    # use the full O(nm) table (still independent of the rolling-row code).
    if len(s) * len(t) <= 4000:
        print(lcs_exhaustive(s, t))
    else:
        print(lcs_table(s, t))


if __name__ == "__main__":
    main()
