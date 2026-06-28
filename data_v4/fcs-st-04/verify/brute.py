#!/usr/bin/env python3
# Independent brute-force oracle for minimum palindromic factorization.
#
# Strategy: O(n^2) DP using a precomputed table isPal[i][j] = "is s[i..j] a
# palindrome" filled by the standard interval DP. dp[i] = min palindromes for
# the prefix s[0..i-1]; dp[0] = 0; dp[j] = min over i<j of dp[i]+1 if s[i..j-1]
# is a palindrome. This is intentionally simple and obviously correct.
#
# Reads the string from stdin (may be empty); prints the minimum count.

import sys


def main():
    data = sys.stdin.read().split()
    s = data[0] if data else ""
    n = len(s)
    if n == 0:
        print(0)
        return

    # isPal[i][j] for 0<=i<=j<n
    isPal = [[False] * n for _ in range(n)]
    for i in range(n):
        isPal[i][i] = True
    for i in range(n - 1):
        isPal[i][i + 1] = (s[i] == s[i + 1])
    for length in range(3, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            isPal[i][j] = (s[i] == s[j]) and isPal[i + 1][j - 1]

    INF = float("inf")
    dp = [INF] * (n + 1)
    dp[0] = 0
    for j in range(1, n + 1):
        for i in range(0, j):
            if dp[i] != INF and isPal[i][j - 1]:
                if dp[i] + 1 < dp[j]:
                    dp[j] = dp[i] + 1
    print(dp[n])


if __name__ == "__main__":
    main()
