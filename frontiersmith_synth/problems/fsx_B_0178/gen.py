#!/usr/bin/env python3
"""gen.py <testId>  ->  ONE 'drone delivery swarm' routing instance on stdout.

Skin of a quantum-circuit transpilation / SWAP-routing task:
  physical nodes  = relay hubs of the swarm's mesh network (the coupling map)
  drones          = logical qubits, drone i starts parked at hub i (identity layout)
  required pairs  = synchronized package hand-offs (two-qubit interactions of one
                    QAOA-style mixing layer) that must be executed while the two
                    drones sit on directly-linked hubs.
testId 1..N is the difficulty ladder (bigger, sparser mesh -> longer routes).
All randomness is seeded solely by testId -> fully deterministic.
"""
import sys, random
from collections import deque


def bfs_dist(adj, s, t, n):
    dist = [-1] * n
    dist[s] = 0
    dq = deque([s])
    while dq:
        u = dq.popleft()
        if u == t:
            return dist[u]
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist[t]


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(9173 + tid)

    n = 6 + tid  # 7 .. 16 for tid 1..10

    # ---- connected sparse mesh (random spanning tree + a few chords) ----
    edges = set()
    for v in range(1, n):
        u = rng.randrange(0, v)
        edges.add(frozenset((u, v)))
    extra = n // 3
    tries = 0
    while extra > 0 and tries < 200:
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a != b and frozenset((a, b)) not in edges:
            edges.add(frozenset((a, b)))
            extra -= 1
        tries += 1

    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = tuple(e)
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        adj[u].sort()

    # ---- required hand-offs: drone pairs at graph distance >= 2 (need routing) ----
    required = []
    seen = set()
    target_k = n
    tries = 0
    while len(required) < target_k and tries < n * n * 8:
        a = rng.randrange(n)
        b = rng.randrange(n)
        key = frozenset((a, b))
        if a != b and key not in seen and bfs_dist(adj, a, b, n) >= 2:
            seen.add(key)
            required.append((a, b))
        tries += 1

    edge_list = sorted(tuple(sorted(tuple(e))) for e in edges)
    out = []
    out.append("%d %d %d" % (n, len(edge_list), len(required)))
    for u, v in edge_list:
        out.append("%d %d" % (u, v))
    for a, b in required:
        out.append("%d %d" % (a, b))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
