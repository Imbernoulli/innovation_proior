#!/usr/bin/env python3
"""Oracle: plain O(n^3) interval DP for the optimal adjacent-merge problem.

We must merge n piles, each merge combining two ADJACENT current piles at a cost
equal to the total weight of the combined pile, until a single pile remains.
Minimize the total merge cost.

dp[i][j] = minimum cost to merge the contiguous block of original piles i..j into
one pile. Try every split point k (i <= k < j): merge i..k into one pile, merge
k+1..j into one pile, then merge those two. The final merge costs the sum of all
weights in i..j, regardless of k.
"""
import sys


def solve(w):
    n = len(w)
    if n <= 1:
        return 0
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + w[i]

    def rng(i, j):  # inclusive sum of w[i..j]
        return pref[j + 1] - pref[i]

    INF = float("inf")
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            best = INF
            for k in range(i, j):
                cand = dp[i][k] + dp[k + 1][j]
                if cand < best:
                    best = cand
            dp[i][j] = best + rng(i, j)
    return dp[0][n - 1]


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    w = [int(x) for x in data[1:1 + n]]
    print(solve(w))


if __name__ == "__main__":
    main()
