#!/usr/bin/env python3
"""Random small-case generator for the cut-vertex path-query problem.

Usage: gen.py SEED

Produces a small undirected graph (possibly with multiple connected components,
parallel edges, and self-loops -- all of which the solver must handle) plus a set
of random queries. Vertex/edge counts are kept tiny so the O(q*n*(n+m)) brute force
stays fast. We deliberately mix:
  * tree-like sparse graphs (lots of articulation points),
  * denser graphs (few/no articulation points),
  * disconnected graphs (to exercise the -1 / unreachable case),
  * parallel edges and self-loops.
Queries include u == v and cross-component pairs.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 9)
    edges = []

    style = rng.random()
    if style < 0.35:
        # Forest / sparse: connect some vertices in a random tree, maybe leave gaps
        # so the graph is disconnected.
        for v in range(2, n + 1):
            if rng.random() < 0.30:
                continue                       # skip -> possibly disconnect
            u = rng.randint(1, v - 1)
            edges.append((u, v))
        extra = rng.randint(0, n)
    elif style < 0.7:
        # Connected-ish: spanning tree then extra edges.
        for v in range(2, n + 1):
            u = rng.randint(1, v - 1)
            edges.append((u, v))
        extra = rng.randint(0, n + 3)
    else:
        # Dense random.
        extra = rng.randint(0, n * 2)

    for _ in range(extra):
        r = rng.random()
        if r < 0.08:
            x = rng.randint(1, n)
            edges.append((x, x))               # self-loop (ignored by solver)
        else:
            u = rng.randint(1, n)
            v = rng.randint(1, n)
            edges.append((u, v))               # may create a parallel edge

    rng.shuffle(edges)
    m = len(edges)

    # Queries.
    q = rng.randint(1, 12)
    queries = []
    for _ in range(q):
        u = rng.randint(1, n)
        v = rng.randint(1, n)
        queries.append((u, v))

    lines = [f"{n} {m}"]
    for (u, v) in edges:
        lines.append(f"{u} {v}")
    lines.append(str(q))
    for (u, v) in queries:
        lines.append(f"{u} {v}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
