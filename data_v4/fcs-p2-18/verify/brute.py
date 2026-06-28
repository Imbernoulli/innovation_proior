#!/usr/bin/env python3
"""
Independent brute-force oracle for: minimum number of character insertions
needed to turn the input string into a palindrome.

This oracle does NOT use the same interval DP as sol.cpp. It uses the
equivalent characterization:

    min insertions = n - LCS(s, reverse(s))

Reasoning: a minimum-insertion palindromization keeps a subsequence of s
that is itself a palindrome (the characters that are NOT inserted, read in
order, form a palindrome), and inserts one character to mirror each of the
remaining characters. The longest palindromic subsequence (LPS) of s equals
LCS(s, reverse(s)). The characters outside the LPS each cost exactly one
insertion, so the answer is n - LPS(s) = n - LCS(s, reverse(s)).

For tiny strings we ALSO run a genuine exhaustive search (try inserting
characters) and cross-check, so the oracle itself is trustworthy.
"""
import sys
from functools import lru_cache


def lcs(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        row = dp[i]
        prow = dp[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                row[j] = prow[j - 1] + 1
            else:
                row[j] = prow[j] if prow[j] >= row[j - 1] else row[j - 1]
    return dp[m][n]


def answer_via_lcs(s: str) -> int:
    return len(s) - lcs(s, s[::-1])


def answer_exhaustive(s: str) -> int:
    """
    True minimal number of insertions, by BFS over the space of strings,
    inserting one character at a time until a palindrome is reached.
    Only safe for very short s (alphabet restricted to chars present in s,
    plus it never helps to insert a char not already in s for *minimality*,
    but to be fully safe we allow inserting any char already in s).
    """
    if s == s[::-1]:
        return 0
    chars = sorted(set(s))
    from collections import deque
    seen = {s}
    q = deque([(s, 0)])
    while q:
        cur, d = q.popleft()
        if cur == cur[::-1]:
            return d
        nd = d + 1
        for pos in range(len(cur) + 1):
            for c in chars:
                nxt = cur[:pos] + c + cur[pos:]
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, nd))
    return 0  # unreachable


def main():
    data = sys.stdin.read().split()
    s = data[0] if data else ""
    val = answer_via_lcs(s)
    # Self-check the oracle on short strings against a genuine exhaustive search.
    if len(s) <= 6:
        ex = answer_exhaustive(s)
        assert ex == val, f"oracle disagreement on {s!r}: exhaustive={ex} lcs={val}"
    print(val)


if __name__ == "__main__":
    main()
