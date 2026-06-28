#!/usr/bin/env python3
# Random small-case generator for "MST with exactly k white edges".
# Usage: python3 gen.py <seed>
# Emits a connected graph (guaranteed via a random spanning tree backbone) plus
# extra random edges, with random colors/weights, and a random k in [0, n-1].
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 7)
    # cap edges so brute (C(m, n-1)) stays cheap
    max_extra = {1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5, 7: 4}[n]
    extra = rng.randint(0, max_extra)

    edges = []
    # spanning-tree backbone to guarantee connectivity (for n >= 2)
    if n >= 2:
        nodes = list(range(n))
        rng.shuffle(nodes)
        for i in range(1, n):
            u = nodes[i]
            v = nodes[rng.randint(0, i - 1)]
            w = rng.randint(0, 9)
            c = rng.randint(0, 1)
            edges.append((u, v, w, c))
    # extra random edges (may be parallel; allowed)
    for _ in range(extra):
        if n == 1:
            break
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        while v == u:
            v = rng.randint(0, n - 1)
        w = rng.randint(0, 9)
        c = rng.randint(0, 1)
        edges.append((u, v, w, c))

    m = len(edges)
    # sometimes ask for an infeasible k to exercise -1 paths
    k = rng.randint(0, max(0, n - 1))
    if rng.random() < 0.15:
        k = rng.randint(0, n + 1)  # may exceed feasible range

    out = [f"{n} {m} {k}"]
    for (u, v, w, c) in edges:
        out.append(f"{u + 1} {v + 1} {w} {c}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
