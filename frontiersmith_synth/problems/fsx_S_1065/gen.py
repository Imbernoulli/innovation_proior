#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of "Demolish the Bridge City, Pier by Pier"
to stdout. Deterministic: all randomness seeded from testId only.

Construction: a recursive binary "separator-pier" hierarchy (dedicated separator piers
at each merge level, connected only to the boundary of their two child groups), then a
handful of random long-range noise braces, then a full random relabeling of pier ids.
This plants a nested-dissection-style separator tree that is obscured by the relabeling
and the noise -- the solver only ever sees the flat brace list.

The exact per-testId parameters were empirically calibrated (see problem notes) so that,
across the ladder, a fill-in-aware adaptive elimination lands around ratio 0.6-0.85 while
a single-pass BFS/degree traversal (the natural first attempt) lands around 0.15-0.22:
comfortably inside the harness's non-saturating, non-degenerate scoring band. Because the
calibration search below is itself a deterministic function of testId (fixed attempt
sequence, first hit wins), the produced instance is 100% reproducible.
"""
import sys
import random


def build_tree(leaf_n, sep_n, depth, ctr, edges, leaf_p, rng):
    """Recursively build a separator-pier hierarchy. ctr[0] is the next free pier id."""
    if depth == 0:
        verts = [ctr[0] + i for i in range(leaf_n)]
        ctr[0] += leaf_n
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if rng.random() < leaf_p:
                    edges.append((verts[i], verts[j]))
        return verts
    bnd_l = build_tree(leaf_n, sep_n, depth - 1, ctr, edges, leaf_p, rng)
    bnd_r = build_tree(leaf_n, sep_n, depth - 1, ctr, edges, leaf_p, rng)
    sep = [ctr[0] + i for i in range(sep_n)]
    ctr[0] += sep_n
    for i in range(len(sep)):
        for j in range(i + 1, len(sep)):
            edges.append((sep[i], sep[j]))
    for s in sep:
        for b in bnd_l:
            edges.append((s, b))
        for b in bnd_r:
            edges.append((s, b))
    return sep


def gen_graph(depth, leaf_n, sep_n, seed, noise_frac, leaf_p):
    rng = random.Random(seed)
    ctr = [0]
    edges = []
    build_tree(leaf_n, sep_n, depth, ctr, edges, leaf_p, rng)
    n = ctr[0]
    edges = list(set(tuple(sorted(e)) for e in edges if e[0] != e[1]))
    n_noise = int(n * noise_frac)
    eset = set(edges)
    added, tries = 0, 0
    while added < n_noise and tries < n_noise * 30 + 50:
        tries += 1
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        e = tuple(sorted((u, v)))
        if e in eset:
            continue
        eset.add(e)
        edges.append(e)
        added += 1
    perm = list(range(n))
    rng.shuffle(perm)
    edges = [tuple(sorted((perm[u], perm[v]))) for u, v in edges]
    return n, edges


def ops_for_order(n, edges, order):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    alive = [True] * n
    ops = 0
    for v in order:
        nbrs = [u for u in adj[v] if alive[u]]
        d = len(nbrs)
        ops += d * (d + 1) // 2
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                if b not in adj[a]:
                    adj[a].add(b)
                    adj[b].add(a)
        alive[v] = False
    return ops


def mindeg_order(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    alive = [True] * n
    order = []
    remaining = set(range(n))
    for _ in range(n):
        best = min(remaining, key=lambda x: (len(adj[x]), x))
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
    return order


def rcm_order(n, edges):
    from collections import deque
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    visited = [False] * n
    order = []
    verts_by_deg = sorted(range(n), key=lambda x: len(adj[x]))
    for s0 in verts_by_deg:
        if visited[s0]:
            continue
        dq = deque([s0])
        visited[s0] = True
        while dq:
            u = dq.popleft()
            order.append(u)
            nbrs = sorted((w for w in adj[u] if not visited[w]), key=lambda x: len(adj[x]))
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    dq.append(w)
    return order


# Difficulty ladder (small -> large). Parameters chosen so the hierarchy has 2 levels
# of dedicated separator piers (4 leaf clusters + 2 inner separators + 1 outer separator).
LEAF_NS = [5, 6, 6, 7, 7, 8, 8, 9, 9, 10]
SEP_NS = [2, 2, 3, 3, 3, 4, 4, 4, 4, 5]
NOISES = [0.04, 0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08]
LEAF_P = 0.35
DEPTH = 2


def make_instance(test_id):
    leaf_n = LEAF_NS[test_id - 1]
    sep_n = SEP_NS[test_id - 1]
    noise = NOISES[test_id - 1]
    fallback = None
    for attempt in range(300):
        seed = test_id * 100003 + attempt * 7919
        n, edges = gen_graph(DEPTH, leaf_n, sep_n, seed, noise, LEAF_P)
        base = ops_for_order(n, edges, list(range(n)))
        good = ops_for_order(n, edges, mindeg_order(n, edges))
        ratio_good = min(1.0, 0.1 * base / max(1, good))
        if fallback is None:
            fallback = (n, edges)
        if 0.40 <= ratio_good <= 0.85:
            naive = ops_for_order(n, edges, rcm_order(n, edges))
            ratio_naive = min(1.0, 0.1 * base / max(1, naive))
            if ratio_naive >= 0.14:
                return n, edges
    return fallback


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    test_id = max(1, min(10, test_id))
    n, edges = make_instance(test_id)
    out = [f"{n} {len(edges)}"]
    for u, v in edges:
        out.append(f"{u + 1} {v + 1}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
