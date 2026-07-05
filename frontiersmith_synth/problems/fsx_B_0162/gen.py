#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance to stdout.

Instance = a QAOA cost-layer routing problem, skinned as coordinating a
solar farm's inverter fleet on a fixed quantum backend coupling map.

testId 1..10 is a difficulty ladder: N (inverter/qubit count) grows and the
hardware graph gets sparser (longer routing distances), so the SWAP budget to
realise every required pairwise interaction grows.
"""
import sys
from collections import deque


def main():
    tid = int(sys.argv[1])
    import random
    rng = random.Random(20260162 + 1000 * tid)

    N = 8 + 2 * (tid - 1)          # 8,10,...,26 physical qubits (= logical qubits)

    # ---- hardware coupling map: random connected sparse graph ----
    nodes = list(range(N))
    rng.shuffle(nodes)
    edges = set()
    for i in range(1, N):          # random spanning tree -> connected
        u = nodes[i]
        v = nodes[rng.randrange(0, i)]
        edges.add((min(u, v), max(u, v)))
    # a few extra edges; fewer for larger tid -> keeps the map sparse (long paths)
    extra = max(1, N // 3 - tid // 3)
    tries = 0
    while extra > 0 and tries < 2000:
        tries += 1
        u = rng.randrange(N)
        v = rng.randrange(N)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if e in edges:
            continue
        edges.add(e)
        extra -= 1
    edges = sorted(edges)

    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def bfs(s):
        d = [-1] * N
        d[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if d[y] < 0:
                    d[y] = d[x] + 1
                    q.append(y)
        return d

    dist = [bfs(s) for s in range(N)]

    # ---- required logical interactions (QAOA cost-layer edges) ----
    K = N + 2 * tid
    all_pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    far = [p for p in all_pairs if dist[p[0]][p[1]] >= 2]
    near = [p for p in all_pairs if dist[p[0]][p[1]] == 1]
    rng.shuffle(far)
    rng.shuffle(near)
    chosen = []
    for p in far:                  # prefer distant pairs -> real routing work
        if len(chosen) >= K:
            break
        chosen.append(p)
    for p in near:
        if len(chosen) >= K:
            break
        chosen.append(p)
    rng.shuffle(chosen)

    out = ["%d %d %d" % (N, len(edges), len(chosen))]
    for u, v in edges:
        out.append("%d %d" % (u, v))
    for a, b in chosen:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
