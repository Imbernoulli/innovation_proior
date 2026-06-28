#!/usr/bin/env python3
"""
Independent brute-force oracle for the house-painting problem.

Min total cost to paint n houses, each one of k colors, no two adjacent houses
sharing a color, costs[i][c] given. Output min cost, or -1 if impossible.

This brute uses a *full* O(n*k*k) DP (no two-minimums trick) so it shares no
optimization logic with the solution under test. For tiny n,k it also falls back
to exhaustive enumeration of all color assignments, giving a second independent
check of the DP itself.
"""
import sys


def solve_full_dp(n, k, costs):
    INF = float("inf")
    if n == 0:
        return 0
    prev = list(costs[0])  # paint house 0 with each color
    for i in range(1, n):
        cur = [INF] * k
        for c in range(k):
            best = INF
            for pc in range(k):
                if pc == c:
                    continue
                if prev[pc] < best:
                    best = prev[pc]
            if best < INF:
                cur[c] = best + costs[i][c]
        prev = cur
    ans = min(prev)
    return -1 if ans == INF else ans


def solve_exhaustive(n, k, costs):
    """Enumerate every coloring; only call for tiny inputs."""
    INF = float("inf")
    best = INF

    def rec(i, last, acc):
        nonlocal best
        if acc >= best:
            return
        if i == n:
            best = min(best, acc)
            return
        for c in range(k):
            if c == last:
                continue
            rec(i + 1, c, acc + costs[i][c])

    if n == 0:
        return 0
    rec(0, -1, 0)
    return -1 if best == INF else best


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    costs = []
    for i in range(n):
        row = []
        for c in range(k):
            row.append(int(data[idx])); idx += 1
        costs.append(row)

    ans = solve_full_dp(n, k, costs)

    # Cross-check with exhaustive enumeration when the state space is tiny.
    if n <= 8 and k <= 5 and k ** n <= 200000:
        ans2 = solve_exhaustive(n, k, costs)
        assert ans == ans2, f"DP/exhaustive disagree: {ans} vs {ans2}"

    print(ans)


if __name__ == "__main__":
    main()
