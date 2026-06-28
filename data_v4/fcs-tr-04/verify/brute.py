#!/usr/bin/env python3
"""Independent O(n^2) oracle: BFS from every node, sum the distances.

Reads the same stdin format as sol.cpp:
  n
  then n-1 lines each "u v" (1-indexed undirected edge)
Prints n lines: for node v (1..n), the sum of dist(v, u) over all u.
"""
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)

    out = []
    for s in range(1, n + 1):
        dist = [-1] * (n + 1)
        dist[s] = 0
        q = deque([s])
        total = 0
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    total += dist[y]
                    q.append(y)
        out.append(str(total))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
