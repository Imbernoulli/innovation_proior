#!/usr/bin/env python3
"""Random SMALL-case generator for the 2-shift (2-coloring) problem.

Usage: python3 gen.py <seed>

Emits a small undirected graph: first line "n m", then m lines "u v".
Mix of guaranteed-bipartite and likely-non-bipartite instances, plus
isolated vertices, multi-edges, and multiple components, so the checker
exercises both POSSIBLE and IMPOSSIBLE branches.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 9)  # keep n small: brute is 2^n
    style = rng.randint(0, 3)

    edges = []
    if style == 0:
        # purely random sparse graph
        max_e = min(12, n * (n - 1) // 2 + 2)
        m = rng.randint(0, max_e)
        for _ in range(m):
            u = rng.randint(1, n)
            v = rng.randint(1, n)
            if u == v:
                continue  # allow drop -> sometimes fewer edges (no self loops)
            edges.append((u, v))
    elif style == 1:
        # bipartition-seeded graph: split into two sides, mostly cross edges
        side = [rng.randint(0, 1) for _ in range(n + 1)]
        cnt = rng.randint(0, 14)
        for _ in range(cnt):
            u = rng.randint(1, n)
            v = rng.randint(1, n)
            if u == v:
                continue
            if rng.random() < 0.85:
                # try to make it a cross edge to stay bipartite-ish
                if side[u] == side[v]:
                    continue
            edges.append((u, v))
    elif style == 2:
        # path or cycle (bipartite iff even cycle)
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        for i in range(n - 1):
            edges.append((perm[i], perm[i + 1]))
        if n >= 3 and rng.random() < 0.5:
            edges.append((perm[-1], perm[0]))  # close the cycle
    else:
        # dense-ish -> often contains an odd cycle / triangle
        max_e = min(16, n * (n - 1) // 2)
        m = rng.randint(0, max_e)
        for _ in range(m):
            u = rng.randint(1, n)
            v = rng.randint(1, n)
            if u == v:
                continue
            edges.append((u, v))

    rng.shuffle(edges)
    print(n, len(edges))
    for (u, v) in edges:
        print(u, v)


if __name__ == "__main__":
    main()
