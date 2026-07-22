# TIER: strong
"""Insight: reformulate "best final load capacity" as a max-flow problem on
the WHOLE candidate graph (deck joints as a super-source, anchors as a
super-sink) -- this is the true theoretical ceiling for the final objective,
and by max-flow/min-cut it is realized by a genuinely REDUNDANT sub-network
(parallel paths sharing the load), not a single spanning tree. Then exploit a
second invariant for the build order: schedule members in non-decreasing
order of hop-distance-from-an-anchor (computed on the chosen sub-network
itself). Under that order, every member touching a joint at hop distance d is
available no later than round d, so by the time any joint accumulates
downstream self-weight demand, ALL of its own supporting members (including
redundant/relief ones) are already built -- support never arrives after the
load it must carry, so the structure never collapses."""
import sys
from collections import deque

INF = 10 ** 9


def max_flow(source, sink, cap):
    flow = 0
    while True:
        parent = {source: None}
        dq = deque([source])
        found = False
        while dq:
            u = dq.popleft()
            if u == sink:
                found = True
                break
            for (a, b), c in list(cap.items()):
                if a == u and c > 0 and b not in parent:
                    parent[b] = u
                    dq.append(b)
        if not found:
            break
        path = []
        v = sink
        while parent[v] is not None:
            u = parent[v]
            path.append((u, v))
            v = u
        bottleneck = min(cap[(u, v)] for (u, v) in path)
        for (u, v) in path:
            cap[(u, v)] -= bottleneck
            cap[(v, u)] = cap.get((v, u), 0) + bottleneck
        flow += bottleneck
    return flow


def main():
    toks = sys.stdin.read().split()
    p = 0
    W, H, M, D = (int(toks[p + i]) for i in range(4)); p += 4
    edges = []
    for _ in range(M):
        u, v, c = int(toks[p]), int(toks[p + 1]), int(toks[p + 2]); p += 3
        edges.append((u, v, c))
    deck = [int(toks[p + i]) for i in range(D)]; p += D
    anchors = set(range(0, W + 1))

    SRC, SNK = "__S__", "__T__"
    cap = {}
    for (u, v, c) in edges:
        cap[(u, v)] = cap.get((u, v), 0) + c
        cap[(v, u)] = cap.get((v, u), 0) + c
    for d in deck:
        cap[(SRC, d)] = cap.get((SRC, d), 0) + INF
    for a in anchors:
        cap[(a, SNK)] = cap.get((a, SNK), 0) + INF

    orig = {}
    for (u, v, c) in edges:
        orig[(u, v)] = orig.get((u, v), 0) + c
        orig[(v, u)] = orig.get((v, u), 0) + c

    max_flow(SRC, SNK, cap)

    chosen = []
    for idx, (u, v, c) in enumerate(edges):
        used = (cap.get((u, v), orig[(u, v)]) < orig[(u, v)]) or \
               (cap.get((v, u), orig[(v, u)]) < orig[(v, u)])
        if used:
            chosen.append(idx)

    # safe order: BFS hop distance from anchors, restricted to the chosen set
    cadj = {}
    for idx in chosen:
        u, v, c = edges[idx]
        cadj.setdefault(u, []).append(v)
        cadj.setdefault(v, []).append(u)
    hop = {}
    dq = deque()
    for a in anchors:
        if a in cadj:
            hop[a] = 0
            dq.append(a)
    while dq:
        u = dq.popleft()
        for v in cadj.get(u, []):
            if v not in hop:
                hop[v] = hop[u] + 1
                dq.append(v)

    def order_key(idx):
        u, v, c = edges[idx]
        du = hop.get(u, 10**9)
        dv = hop.get(v, 10**9)
        return (min(du, dv), max(du, dv), idx)

    chosen.sort(key=order_key)

    out = [str(len(chosen))]
    out.extend(str(i) for i in chosen)
    print("\n".join(out))


if __name__ == "__main__":
    main()
