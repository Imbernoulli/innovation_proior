# TIER: strong
"""Two composed insights, matching the two things the brace pattern hides:

1. separator recovery -- before ever choosing a demolition order, do a cheap
   recursive spectral bisection of the brace graph (power iteration on a shifted
   Laplacian, deflating the constant vector, to approximate the Fiedler vector) to
   recover a coarse "which side of the hidden hierarchy is this pier probably on"
   label for every pier. This works on the RELABELED graph itself -- it needs no
   knowledge of how the piers were originally grouped.

2. fill-in-aware ordering -- demolish piers with an ADAPTIVE (re-evaluated after
   every removal) minimum-current-degree rule, so it always reacts to the actual
   live brace pattern including whatever cross-braces earlier removals installed.
   Ties (equal current degree, which are common right after a merge) are broken
   using the recovered coarse label, preferring piers that look "deeper" in the
   recovered hierarchy (i.e. more nested / more leaf-like) over piers that look
   like recovered separators, so that likely separator piers are naturally
   deferred until the sides they join have mostly been cleared.

Neither piece alone reliably beats a plain non-adaptive sweep; together they let
the order track the true (recovered) hierarchy without ever seeing it directly.
"""
import sys
import random
from collections import deque


def bfs_dist(vset, adj, src):
    dist = {src: 0}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if w in vset and w not in dist:
                dist[w] = dist[u] + 1
                dq.append(w)
    return dist


def fiedler_vector(vs, vset, adj, iters=25, seed=12345):
    idx = {v: i for i, v in enumerate(vs)}
    m = len(vs)
    deg = [len(adj[v] & vset) for v in vs]
    dmax = max(deg) if deg else 1
    c = 2.0 * dmax + 1.0
    rng = random.Random(seed)
    x = [rng.uniform(-1, 1) for _ in range(m)]

    def deflate(v):
        mean = sum(v) / m
        v = [vi - mean for vi in v]
        norm = sum(vi * vi for vi in v) ** 0.5 or 1.0
        return [vi / norm for vi in v]

    x = deflate(x)
    nbr_idx = [[idx[w] for w in adj[v] if w in vset] for v in vs]
    for _ in range(iters):
        y = [0.0] * m
        for i in range(m):
            s = 0.0
            for j in nbr_idx[i]:
                s += x[j]
            y[i] = (c - deg[i]) * x[i] + s
        x = deflate(y)
    return {vs[i]: x[i] for i in range(m)}


def coarse_labels(n, adj, max_depth=3, leaf_size=6):
    label = [0] * n

    def recurse(vs, depth, prefix):
        vset = set(vs)
        for v in vset:
            label[v] = prefix
        if depth >= max_depth or len(vset) <= leaf_size:
            return
        start = next(iter(vset))
        dist = bfs_dist(vset, adj, start)
        if len(dist) < len(vset):
            reached = set(dist.keys())
            unreached = vset - reached
            recurse(reached, depth + 1, prefix * 2)
            recurse(unreached, depth + 1, prefix * 2 + 1)
            return
        vs_sorted = sorted(vset)
        fv = fiedler_vector(vs_sorted, vset, adj)
        ordv = sorted(vs_sorted, key=lambda v: (fv[v], v))
        mid = len(ordv) // 2
        left, right = set(ordv[:mid]), set(ordv[mid:])
        if not left or not right:
            return
        recurse(left, depth + 1, prefix * 2)
        recurse(right, depth + 1, prefix * 2 + 1)

    recurse(set(range(n)), 0, 1)
    return label


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    adj = [set() for _ in range(n)]
    for _ in range(m):
        u = int(data[idx]) - 1; idx += 1
        v = int(data[idx]) - 1; idx += 1
        adj[u].add(v)
        adj[v].add(u)

    label = coarse_labels(n, adj)

    alive = [True] * n
    order = []
    remaining = set(range(n))
    for _ in range(n):
        best = min(remaining, key=lambda x: (len(adj[x]), -label[x], x))
        order.append(best)
        nbrs = [u for u in adj[best] if alive[u]]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                adj[a].add(b)
                adj[b].add(a)
        alive[best] = False
        remaining.discard(best)
        for u in nbrs:
            adj[u].discard(best)

    print(" ".join(str(v + 1) for v in order))


if __name__ == "__main__":
    main()
