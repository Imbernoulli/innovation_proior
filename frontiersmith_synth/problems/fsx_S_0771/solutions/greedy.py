# TIER: greedy
"""The obvious first attempt: connect deck to anchors with a single
maximum-capacity spanning tree (Kruskal, widest-path thinking -- the textbook
answer to "which members should I use for capacity?"), and build it breadth-
first from the ground so at least the ORDER is sane. What it never considers
is that a tree can carry, at any cut, only as much as its ONE surviving path
-- it has no notion of splitting a joint's load across several members at
once. When the candidate graph offers several independently-capacitated
routes to the ground, a tree is structurally forced to pick (essentially)
just one of them and leaves the rest of the capacity on the table."""
import sys
from collections import deque


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


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
    n = (W + 1) * (H + 1)

    order_by_cap = sorted(range(M), key=lambda i: (-edges[i][2], i))

    dsu = DSU(n)

    chosen = []
    for idx in order_by_cap:
        u, v, c = edges[idx]
        if dsu.union(u, v):
            chosen.append(idx)

    # build breadth-first from the ground (never leaves a joint stranded),
    # tie-broken by "reinforce with the strongest member first"
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
        return (min(du, dv), max(du, dv), -c, idx)

    chosen.sort(key=order_key)

    out = [str(len(chosen))]
    out.extend(str(i) for i in chosen)
    print("\n".join(out))


if __name__ == "__main__":
    main()
