# TIER: strong
"""Covering-space insight: low-order closed-walk moments are purely LOCAL
observables (a walk of length <=k can only explore a bounded neighborhood).
So spend the entire vertex budget building an exact covering LIFT of the
template's unique cycle instead of copying-and-random-wiring:

  1. Find H's one cycle (break it at edge e*=(a,b); the rest is a
     spanning tree T of H).
  2. Take L = n // nH copies of T (an exact clone each -- degrees, and
     hence every local walk-based statistic, are reproduced exactly).
  3. Re-glue the copies with the special edge e* into ONE big necklace:
     copy i's 'b' connects to copy (i+1 mod L)'s 'a'.

This is a genuine graph covering map of H. Every vertex keeps EXACTLY its
H-degree (so the degree-square-sum invariant matches exactly), and any
closed walk of length < L * girth(H) in the lift corresponds bijectively
to a closed walk in H with zero net winding around the cycle -- so the
walk moments match (exactly, or within the tiny fixed leakage from walks
that do wind once around H's own short cycle). All the freedom the
audit does not see goes into the *global* topology: the lift's girth (and
hence its diameter) grows LINEARLY in L, unlike a tree-of-copies whose
diameter only grows like log(L)."""
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

    # find the unique cycle: a spanning tree covers nH-1 edges, the single
    # remaining edge is estar. Build a tree via BFS/union-find on the fly.
    parent = list(range(nH))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    estar = None
    tree_edges = []
    for (u, v) in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            estar = (u, v)
        else:
            parent[ru] = rv
            tree_edges.append((u, v))
    a, b = estar

    L = max(1, n // nH)
    N = nH * L

    def gidx(i, v):
        return i * nH + v

    out_edges = []
    for i in range(L):
        for (u, v) in tree_edges:
            out_edges.append((gidx(i, u), gidx(i, v)))
    for i in range(L):
        j = (i + 1) % L
        out_edges.append((gidx(i, b), gidx(j, a)))

    out = [str(N), str(len(out_edges))]
    for (u, v) in out_edges:
        out.append(f"{u} {v}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
