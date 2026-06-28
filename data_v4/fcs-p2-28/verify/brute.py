#!/usr/bin/env python3
"""Independent brute oracle for Longest Palindromic Subsequence.

Strategy: longest palindromic subsequence of s equals the LCS of s and its
reverse. We compute LCS(s, reverse(s)) with the standard O(n^2) DP. This is a
completely different formulation from the interval DP in sol.cpp, so it serves
as an independent check. For tiny strings we ALSO brute-force by trying every
subsequence (mask enumeration) and assert agreement, giving a third check.
"""
import sys


def lcs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = dp[i]
        prow = dp[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                row[j] = prow[j - 1] + 1
            else:
                row[j] = prow[j] if prow[j] >= row[j - 1] else row[j - 1]
    return dp[n][m]


def lps_via_lcs(s):
    return lcs(s, s[::-1])


def is_pal(t):
    return t == t[::-1]


def lps_bruteforce(s):
    n = len(s)
    best = 0
    for mask in range(1 << n):
        sub = ''.join(s[i] for i in range(n) if (mask >> i) & 1)
        if is_pal(sub) and len(sub) > best:
            best = len(sub)
    return best


def solve(s):
    val = lps_via_lcs(s)
    if len(s) <= 18:  # cross-check the two independent methods on small inputs
        bf = lps_bruteforce(s)
        assert val == bf, f"oracle internal mismatch on {s!r}: lcs={val} bf={bf}"
    return val


def main():
    data = sys.stdin.read().split()
    s = data[0] if data else ""
    print(solve(s))


if __name__ == "__main__":
    main()
