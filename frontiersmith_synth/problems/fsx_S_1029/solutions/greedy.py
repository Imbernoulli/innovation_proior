# TIER: greedy
"""The 'obvious' recipe: replicate the template many times to fill the
vertex budget, and wire the copies into a tree (via one designated leaf
vertex per copy) so the whole thing stays connected.  This preserves the
audited local statistics reasonably well (each copy is an exact clone of H,
only O(n/nH) leaf vertices get a small degree bump), so it PASSES the
audit -- but a tree-of-copies has diameter only O(depth) ~ O(log(n/nH)):
building more copies does not translate into a proportionally larger
diameter.  No covering-space idea is used here at all."""
import sys
from collections import defaultdict


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    nH, k, n = map(int, data[idx].split()); idx += 1
    idx += 1  # eps
    idx += 1  # eps2
    idx += 1  # D_MAX M_MAX
    mH = int(data[idx]); idx += 1
    edges = []
    adjH = defaultdict(list)
    for _ in range(mH):
        u, v = map(int, data[idx].split()); idx += 1
        edges.append((u, v))
        adjH[u].append(v)
        adjH[v].append(u)

    degs = [len(adjH[v]) for v in range(nH)]
    leaves = [v for v in range(nH) if degs[v] == 1]
    leaf = leaves[0] if leaves else min(range(nH), key=lambda v: degs[v])

    C = max(1, n // nH)
    N = nH * C

    def gidx(i, v):
        return i * nH + v

    out_edges = []
    for i in range(C):
        for (u, v) in edges:
            out_edges.append((gidx(i, u), gidx(i, v)))
    for i in range(1, C):
        p = (i - 1) // 2
        out_edges.append((gidx(i, leaf), gidx(p, leaf)))

    out = [str(N), str(len(out_edges))]
    for (u, v) in out_edges:
        out.append(f"{u} {v}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
