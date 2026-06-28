#!/usr/bin/env python3
"""Independent brute-force oracle for fcs-tr-02.

Count unordered pairs (u, v), u != v, such that the sum of edge weights on the
tree path between u and v equals exactly L. Method: BFS from every vertex,
accumulate exact-L hits, divide by 2. O(n^2) — only for small n.
"""
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        w = int(data[idx]); idx += 1
        adj[u].append((v, w))
        adj[v].append((u, w))

    total = 0
    for s in range(1, n + 1):
        dist = [-1] * (n + 1)
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for (v, w) in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + w
                    dq.append(v)
        for t in range(1, n + 1):
            if t != s and dist[t] == L:
                total += 1
    print(total // 2)


if __name__ == "__main__":
    main()
