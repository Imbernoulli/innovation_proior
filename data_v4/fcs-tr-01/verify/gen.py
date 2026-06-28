#!/usr/bin/env python3
"""
Random small-case generator. Usage: gen.py <seed> [maxn] [maxq]
Prints a valid test in the problem's input format.

Each query is a non-empty set of distinct important vertices. Note: the problem
GUARANTEES sum of k over queries <= some bound; for stress we keep n small.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    maxq = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    rng = random.Random(seed)

    n = rng.randint(1, maxn)
    edges = []
    for v in range(2, n + 1):
        u = rng.randint(1, v - 1)  # random rooted tree -> connected
        edges.append((u, v))
    # shuffle endpoints/order so the input is not in a canonical orientation
    out = [str(n)]
    rng.shuffle(edges)
    for (u, v) in edges:
        if rng.random() < 0.5:
            u, v = v, u
        out.append(f"{u} {v}")

    q = rng.randint(1, maxq)
    out.append(str(q))
    for _ in range(q):
        k = rng.randint(1, n)
        verts = rng.sample(range(1, n + 1), k)
        out.append(str(len(verts)) + " " + " ".join(map(str, verts)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
