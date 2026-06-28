#!/usr/bin/env python3
"""Brute-force oracle for the cut-vertex path-query problem.

For each query (u, v):
  * if u == v: answer 0.
  * if u and v are in different connected components: answer -1 (no path).
  * otherwise: count the number of vertices w (w != u, w != v) such that REMOVING w
    from the graph disconnects u from v. This is exactly the set of articulation
    points that lie on every u-v path.

Method: directly try removing each candidate vertex and re-test u-v reachability by
BFS. O(q * n * (n + m)) -- only used on small cases.

Reads stdin in the same format as sol.cpp:
    n m
    m lines: u v
    q
    q lines: u v
"""
import sys
from collections import deque


def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    adj = [set() for _ in range(n + 1)]
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)

    def reachable(src, dst, banned):
        if src == banned or dst == banned:
            return False
        seen = [False] * (n + 1)
        seen[src] = True
        dq = deque([src])
        while dq:
            x = dq.popleft()
            if x == dst:
                return True
            for y in adj[x]:
                if y == banned or seen[y]:
                    continue
                seen[y] = True
                dq.append(y)
        return seen[dst]

    # Connected-component ids (for the -1 / unreachable case).
    comp = [0] * (n + 1)
    c = 0
    for s in range(1, n + 1):
        if comp[s]:
            continue
        c += 1
        comp[s] = c
        dq = deque([s])
        while dq:
            x = dq.popleft()
            for y in adj[x]:
                if not comp[y]:
                    comp[y] = c
                    dq.append(y)

    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        if u == v:
            out.append("0")
            continue
        if comp[u] != comp[v]:
            out.append("-1")
            continue
        cnt = 0
        for w in range(1, n + 1):
            if w == u or w == v:
                continue
            # w separates u from v iff u and v become unreachable without w.
            if not reachable(u, v, w):
                cnt += 1
        out.append(str(cnt))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
