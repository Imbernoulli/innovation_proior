#!/usr/bin/env python3
# Random small-case generator. Param: seed (int).
# Produces graphs small enough for the brute force (m <= ~14).
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed)

    n = rnd.randint(1, 6)          # vertices
    m = rnd.randint(0, 13)         # edges (brute is 2^m)
    K = rnd.randint(1, 4)          # colors

    # capacities: range from 0 up to m (sometimes tight, sometimes loose)
    caps = []
    for c in range(K):
        # bias toward small caps to exercise the partition matroid
        caps.append(rnd.randint(0, max(1, m)))

    edges = []
    for _ in range(m):
        u = rnd.randint(1, n)
        v = rnd.randint(1, n)
        # allow self-loops occasionally and parallel edges naturally
        col = rnd.randint(1, K)
        edges.append((u, v, col))

    out = []
    out.append(f"{n} {m} {K}")
    out.append(" ".join(str(x) for x in caps))
    for (u, v, col) in edges:
        out.append(f"{u} {v} {col}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
