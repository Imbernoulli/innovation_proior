#!/usr/bin/env python3
"""Random small-case generator for fcs-tr-02.

Usage: python3 gen.py <seed>
Emits a random weighted tree and a target L.
Small parameters so the O(n^2) brute force stays fast.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 12)          # small trees, including n = 1
    maxw = rng.choice([1, 2, 3, 5]) # small weights so distances stay tiny
    # L chosen near the achievable range so many cases have nonzero answers,
    # but also sometimes 0 or unreachable.
    L = rng.randint(0, n * maxw + 2)

    edges = []
    for v in range(2, n + 1):
        u = rng.randint(1, v - 1)   # random parent -> random tree shape
        w = rng.randint(1, maxw)    # positive weights
        edges.append((u, v, w))

    # Randomly relabel vertices to avoid any positional bias.
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    relabel = {i + 1: perm[i] for i in range(n)}

    out = [f"{n} {L}"]
    # shuffle edge order and endpoint order too
    shuffled = edges[:]
    rng.shuffle(shuffled)
    for (u, v, w) in shuffled:
        a, b = relabel[u], relabel[v]
        if rng.random() < 0.5:
            a, b = b, a
        out.append(f"{a} {b} {w}")

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
