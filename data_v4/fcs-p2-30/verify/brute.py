#!/usr/bin/env python3
"""
Independent brute-force oracle for the digit-string decoding count problem.

Contract (matches context.md):
  stdin:  line 1 = prime p, line 2 = digit string s (length >= 1)
  stdout: number of ways to decode s under the map 1->A .. 26->Z, taken mod p.

This oracle uses an entirely separate formulation from sol.cpp: it counts decodings
by literal recursion / enumeration over the FIRST few characters for short strings
(true exhaustive search, no DP recurrence reused), and falls back to an
arbitrary-precision dp (computed WITHOUT taking the modulus until the very end) for
longer strings so it can still serve as a reference on bigger random tests. Both
paths compute the same mathematical quantity by different means.
"""
import sys


def count_exhaustive(s: str) -> int:
    """True exhaustive enumeration: try every way to cut s into 1- and 2-digit groups,
    each group being a valid letter code (1..9 for one digit, 10..26 for two digits).
    Exponential; only used for short strings as an independent check."""
    n = len(s)

    def rec(i: int) -> int:
        if i == n:
            return 1
        total = 0
        # one-digit group
        if s[i] != '0':
            total += rec(i + 1)
        # two-digit group
        if i + 1 < n:
            v = int(s[i:i + 2])
            if 10 <= v <= 26:
                total += rec(i + 2)
        return total

    return rec(0)


def count_bigint_dp(s: str) -> int:
    """Reference dp in Python big integers (exact, no modulus until the caller reduces).
    Independent of sol.cpp's rolling-window form; here we keep the full array."""
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        if s[i - 1] != '0':
            dp[i] += dp[i - 1]
        if i >= 2:
            v = int(s[i - 2:i])
            if 10 <= v <= 26:
                dp[i] += dp[i - 2]
    return dp[n]


def solve(p: int, s: str) -> int:
    if len(s) <= 18:
        return count_exhaustive(s) % p
    return count_bigint_dp(s) % p


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    p = int(data[0])
    s = data[1] if len(data) > 1 else ""
    print(solve(p, s))


if __name__ == "__main__":
    main()
