#!/usr/bin/env python3
# Independent O(p * n^2) DP oracle for the mailbox placement problem.
# dp[k][i] = min total distance to serve the first i houses using exactly k mailboxes.
# cost(j, i) = sum of |x[t] - median| for t in [j, i], one mailbox optimally placed
# to serve the contiguous block of houses x[j..i].
# Intentionally slow and obviously correct: cost(j,i) is computed by an explicit
# scan to the median, and dp by an explicit O(n) inner loop over split points.
import sys


def solve(data):
    it = iter(data)
    n = next(it)
    p = next(it)
    x = sorted(next(it) for _ in range(n))

    INF = float("inf")

    # Precompute cost(j, i) for 0 <= j <= i < n by explicit scan to the median.
    cost = [[0] * n for _ in range(n)]
    for j in range(n):
        for i in range(j, n):
            med = x[(j + i) // 2]  # lower median minimizes sum of abs deviations
            cost[j][i] = sum(abs(x[t] - med) for t in range(j, i + 1))

    # dp[k][i] = min cost to cover houses 0..i-1 with exactly k mailboxes.
    dp = [[INF] * (n + 1) for _ in range(p + 1)]
    dp[0][0] = 0
    for k in range(1, p + 1):
        for i in range(k, n + 1):  # need at least k houses for k non-empty groups
            best = INF
            for j in range(k - 1, i):  # last group = houses [j .. i-1]
                if dp[k - 1][j] == INF:
                    continue
                cand = dp[k - 1][j] + cost[j][i - 1]
                if cand < best:
                    best = cand
            dp[k][i] = best

    # Using more mailboxes never increases cost, so "exactly p" with p <= n is the
    # value we want; guaranteed p <= n by constraints.
    return dp[p][n]


def main():
    data = list(map(int, sys.stdin.read().split()))
    print(solve(data))


if __name__ == "__main__":
    main()
