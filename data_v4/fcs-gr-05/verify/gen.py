#!/usr/bin/env python3
"""Random small-case generator for 2-edge-connectivity augmentation.

Usage: gen.py SEED

Produces a CONNECTED undirected graph (the problem guarantees connectivity).
We first lay down a random spanning tree to guarantee connectivity, then add a
random number of extra edges (allowing parallel edges and self-loops, both of
which the solver must handle: parallel edges are never bridges, self-loops are
ignored). Vertex count is small so the brute force stays tractable.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 9)
    edges = []

    # Random spanning tree -> guarantees the graph is connected.
    for v in range(2, n + 1):
        u = rng.randint(1, v - 1)
        edges.append((u, v))

    # Extra edges: random pairs, sometimes parallel, sometimes self-loops.
    extra = rng.randint(0, n + 2)
    for _ in range(extra):
        r = rng.random()
        if r < 0.10 and n >= 1:
            x = rng.randint(1, n)
            edges.append((x, x))            # self-loop
        else:
            u = rng.randint(1, n)
            v = rng.randint(1, n)
            edges.append((u, v))            # may duplicate an existing edge

    rng.shuffle(edges)
    m = len(edges)

    out = [f"{n} {m}"]
    for (u, v) in edges:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
