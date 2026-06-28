#!/usr/bin/env python3
# Random small-case generator for "subtree distinct colors".
# Usage: gen.py <seed>
# Prints: n, then n colors, then n-1 edges (1-indexed). Tree rooted at node 1.
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of sizes; bias toward small but include n=1 and a few bigger.
    r = rng.random()
    if r < 0.10:
        n = 1
    elif r < 0.25:
        n = rng.randint(2, 4)
    elif r < 0.85:
        n = rng.randint(2, 20)
    else:
        n = rng.randint(20, 60)

    # Colors: sometimes few distinct (forces collisions), sometimes wide range,
    # sometimes large arbitrary values (tests coordinate compression).
    mode = rng.randint(0, 3)
    if mode == 0:
        maxc = 1                      # all same color
    elif mode == 1:
        maxc = max(1, n // 3)
    elif mode == 2:
        maxc = 10 ** 9                # wide / arbitrary
    else:
        maxc = n
    colors = [rng.randint(1, maxc) for _ in range(n)]

    # Random tree: attach each new node to a random earlier node.
    # Sometimes make it a path or a star to hit structural edges.
    shape = rng.randint(0, 2)
    edges = []
    for v in range(2, n + 1):  # nodes 2..n (1-indexed); node 1 is root
        if shape == 0:
            p = v - 1            # path
        elif shape == 1:
            p = 1                # star
        else:
            p = rng.randint(1, v - 1)  # random
        edges.append((p, v))

    # Shuffle edge endpoint order and edge order to avoid any input ordering assumption.
    out = [str(n), " ".join(map(str, colors))]
    rng.shuffle(edges)
    for (a, b) in edges:
        if rng.random() < 0.5:
            a, b = b, a
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
