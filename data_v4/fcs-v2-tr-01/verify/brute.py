#!/usr/bin/env python3
# Independent oracle for Cut-to-Quarantine.
#
# The minimum cost to cut edges so that no special node stays connected to the root
# is exactly a min s-t cut: put the root as source, every special node as sink, each
# tree edge as an undirected edge of capacity = its cut cost. By max-flow / min-cut
# duality the min cut equals the max flow. We compute max flow with a plain
# Edmonds-Karp BFS augmentation (obviously correct, slow). This is structurally
# unrelated to the virtual-tree DP in sol.cpp.

import sys
from collections import deque


def solve():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    edges = []
    for _ in range(n - 1):
        u = int(data[idx]); v = int(data[idx + 1]); w = int(data[idx + 2]); idx += 3
        edges.append((u, v, w))
    q = int(data[idx]); idx += 1
    out = []

    # Super-source = 0 (we attach the root, node 1, as source directly).
    # Nodes are 1..n; sink super-node = n+1.
    SINK = n + 1
    N = n + 2  # 0 unused, 1..n nodes, n+1 sink

    for _ in range(q):
        k = int(data[idx]); idx += 1
        spec = [int(data[idx + i]) for i in range(k)]
        idx += k

        # Build a flow network: tree edges (undirected -> two directed arcs each with
        # capacity w), plus an infinite-capacity arc from each special node to SINK.
        # Source = node 1 (root). Min cut(source, sink) = answer.
        INF = float('inf')

        # adjacency as list of [to, cap, rev_index]
        graph = [[] for _ in range(N)]

        def add_edge(a, b, cap):
            graph[a].append([b, cap, len(graph[b])])
            graph[b].append([a, 0, len(graph[a]) - 1])

        for (u, v, w) in edges:
            # undirected capacity w both directions: add two arcs each cap w sharing
            # nothing, by adding edge u->v cap w and v->u cap w as separate forward arcs.
            add_edge(u, v, w)
            add_edge(v, u, w)

        for s in spec:
            add_edge(s, SINK, INF)

        source = 1
        flow = 0
        while True:
            # BFS to find augmenting path, track parent edges
            parent = [(-1, -1)] * N
            parent[source] = (source, -1)
            dq = deque([source])
            while dq:
                x = dq.popleft()
                if x == SINK:
                    break
                for ei, (to, cap, rev) in enumerate(graph[x]):
                    if cap > 0 and parent[to][0] == -1:
                        parent[to] = (x, ei)
                        dq.append(to)
            if parent[SINK][0] == -1:
                break
            # bottleneck
            bott = INF
            cur = SINK
            while cur != source:
                px, pei = parent[cur]
                bott = min(bott, graph[px][pei][1])
                cur = px
            cur = SINK
            while cur != source:
                px, pei = parent[cur]
                graph[px][pei][1] -= bott
                rev = graph[px][pei][2]
                graph[cur][rev][1] += bott
                cur = px
            flow += bott

        out.append(str(int(flow)))

    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    solve()
