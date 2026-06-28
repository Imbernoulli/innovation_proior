#!/usr/bin/env python3
# Random small tree generator. Usage: gen.py <seed>
# Emits: n, then n-1 edges (random tree, random labels/order).
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # bias toward small n, with occasional n=1, paths, and stars
    shape = rng.randint(0, 4)
    n = rng.randint(1, 9)
    edges = []
    if n >= 2:
        if shape == 1:
            # path
            perm = list(range(1, n + 1))
            rng.shuffle(perm)
            for i in range(1, n):
                edges.append((perm[i - 1], perm[i]))
        elif shape == 2:
            # star
            center = rng.randint(1, n)
            for v in range(1, n + 1):
                if v != center:
                    edges.append((center, v))
        else:
            # generic random tree via random parent attachment
            perm = list(range(1, n + 1))
            rng.shuffle(perm)
            for i in range(1, n):
                p = perm[rng.randint(0, i - 1)]
                edges.append((p, perm[i]))
    # randomize edge orientation and order
    rng.shuffle(edges)
    edges = [(u, v) if rng.random() < 0.5 else (v, u) for (u, v) in edges]

    lines = [str(n)]
    for (u, v) in edges:
        lines.append(f"{u} {v}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
