#!/usr/bin/env python3
"""gen.py <testId>  -> prints ONE routing instance to stdout.

Instance schema (all 0-based):
  line 1: n m k
  next m lines: "u v"      undirected coupling-map edges over n physical slots
  next 1 line: n ints      init[p] = logical debris-tug on physical slot p (a permutation)
  next k lines: "a b"      interaction between logical tugs a and b (a joint capture maneuver)

Difficulty ladder testId 1..10: bigger grids, more maneuvers, permuted initial placement.
Deterministic: seeded only by testId.
"""
import sys
from collections import deque


def shortest_path(adj, s, t):
    if s == t:
        return [s]
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in prev:
                prev[v] = u
                if v == t:
                    path = [t]
                    while path[-1] != s:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(v)
    return None


def baseline_swaps(n, adj, init, pairs):
    pos = list(init)
    inv = [0] * n
    for p in range(n):
        inv[pos[p]] = p
    sw = 0
    for (a, b) in pairs:
        pa = inv[a]
        pb = inv[b]
        path = shortest_path(adj, pa, pb)
        for i in range(len(path) - 2):
            u = path[i]
            v = path[i + 1]
            sw += 1
            lu = pos[u]
            lv = pos[v]
            pos[u] = lv
            pos[v] = lu
            inv[lv] = u
            inv[lu] = v
    return sw


def build_grid(R, C):
    n = R * C
    edges = []
    adj = [set() for _ in range(n)]
    for r in range(R):
        for c in range(C):
            u = r * C + c
            if c + 1 < C:
                v = r * C + (c + 1)
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
            if r + 1 < R:
                v = (r + 1) * C + c
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
    return n, edges, adj


# ladder: (rows, cols, k, permute)
LADDER = {
    1: (2, 3, 4, False),
    2: (2, 3, 5, True),
    3: (3, 3, 6, True),
    4: (3, 3, 8, True),
    5: (3, 4, 9, True),
    6: (3, 4, 12, True),
    7: (4, 4, 12, True),
    8: (4, 4, 16, True),
    9: (4, 5, 18, True),
    10: (4, 5, 22, True),
}


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid not in LADDER:
        tid = ((tid - 1) % 10) + 1
    import random
    rng = random.Random(1000 + tid)

    R, C, k, permute = LADDER[tid]
    n, edges, adj = build_grid(R, C)

    if permute:
        init = list(range(n))
        rng.shuffle(init)
    else:
        init = list(range(n))

    # sample k distinct unordered logical pairs (the QAOA-style interaction graph)
    all_pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(all_pairs)
    pairs = all_pairs[:k]

    # guarantee a positive routing baseline: at least one non-adjacent maneuver.
    if baseline_swaps(n, adj, init, pairs) == 0:
        # replace last pair with a maximally distant logical pair
        best = None
        bestd = -1
        inv = [0] * n
        for p in range(n):
            inv[init[p]] = p
        for a in range(n):
            for b in range(a + 1, n):
                d = len(shortest_path(adj, inv[a], inv[b])) - 1
                if d > bestd:
                    bestd = d
                    best = (a, b)
        pairs[-1] = best

    out = []
    out.append("%d %d %d" % (n, len(edges), k))
    for (u, v) in edges:
        out.append("%d %d" % (u, v))
    out.append(" ".join(str(x) for x in init))
    for (a, b) in pairs:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
