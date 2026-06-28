#!/usr/bin/env python3
"""Independent brute oracle for the minimum-coins problem.

Approach: BFS over reachable sums, where each "level" of the BFS corresponds to
using exactly k coins. The first time we reach the target S, the level k is the
minimum number of coins. This is a fundamentally different formulation from the
forward value-DP in sol.cpp (BFS-by-coin-count vs. relax-each-sum), which makes
it a genuine cross-check rather than a re-implementation.

Reads the same stdin format as sol.cpp:
    n S
    c[0] c[1] ... c[n-1]
Prints the minimum number of coins, or -1 if S cannot be formed.
"""
import sys
from collections import deque


def solve(data):
    it = iter(data.split())
    try:
        n = int(next(it))
        S = int(next(it))
    except StopIteration:
        return None
    coins = [int(next(it)) for _ in range(n)]

    if S == 0:
        return 0

    # Keep only distinct positive denominations <= S; others can never help.
    useful = sorted(set(c for c in coins if 0 < c <= S))
    if not useful:
        return -1

    # BFS over sums; dist[s] = min coins to reach s.
    dist = [-1] * (S + 1)
    dist[0] = 0
    q = deque([0])
    while q:
        s = q.popleft()
        if s == S:
            return dist[s]
        d = dist[s]
        for c in useful:
            t = s + c
            if t <= S and dist[t] == -1:
                dist[t] = d + 1
                if t == S:
                    return dist[t]
                q.append(t)
    return -1 if dist[S] == -1 else dist[S]


def main():
    data = sys.stdin.read()
    res = solve(data)
    if res is None:
        return
    print(res)


if __name__ == "__main__":
    main()
