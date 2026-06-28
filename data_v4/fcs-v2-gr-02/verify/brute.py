#!/usr/bin/env python3
import sys
from collections import deque

def reachable(adj, n, s, banned):
    """Set of vertices reachable from s in the graph with vertex `banned` deleted."""
    seen = [False] * (n + 1)
    if banned == s:
        return seen
    dq = deque([s])
    seen[s] = True
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v == banned:
                continue
            if not seen[v]:
                seen[v] = True
                dq.append(v)
    return seen

def solve(data):
    it = iter(data)
    n = int(next(it)); m = int(next(it)); s = int(next(it))
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        a = int(next(it)); b = int(next(it))
        adj[a].append(b)

    base = reachable(adj, n, s, 0)  # banned=0 deletes nothing (vertices are 1..n)
    R = [v for v in range(1, n + 1) if base[v]]  # reachable set including s

    # dom[v] = set of vertices that dominate v (every s->v path passes through them),
    # for each reachable v. v always dominates itself; s dominates everything reachable.
    dom = {v: set() for v in R}
    for u in range(1, n + 1):
        if u == s or not base[u]:
            continue
        after = reachable(adj, n, s, u)
        for v in R:
            if v == u:
                continue
            # v reachable before, unreachable after removing u => u dominates v
            if base[v] and not after[v]:
                dom[v].add(u)
    for v in R:
        dom[v].add(v)   # self
        dom[v].add(s)   # source dominates all reachable

    idom = [0] * (n + 1)
    for v in R:
        if v == s:
            idom[v] = 0
            continue
        proper = dom[v] - {v}            # proper dominators
        # idom(v) is the proper dominator dominated by every other proper dominator:
        # i.e. the one with the largest dominator set among `proper`.
        best = None
        for u in proper:
            # u is idom candidate if every other proper dominator d dominates u
            ok = True
            for d in proper:
                if d == u:
                    continue
                if d not in dom[u]:
                    ok = False
                    break
            if ok:
                best = u
                break
        idom[v] = best if best is not None else s
    return idom[1:n + 1]

def main():
    data = sys.stdin.read().split()
    res = solve(data)
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()
