#!/usr/bin/env python3
"""Random small-tree generator. Usage: gen.py <seed>

Emits a valid tree in the problem's stdin format:
  n
  n-1 lines of "u v" (1-indexed), a uniformly-ish random labelled tree with
  edges shuffled and endpoints randomly oriented so sol.cpp must not rely on
  any ordering or rooting convention.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of shapes/sizes; bias toward small so brute is fast but include n=1,2.
    r = rng.random()
    if r < 0.08:
        n = 1
    elif r < 0.16:
        n = 2
    else:
        n = rng.randint(3, 60)

    edges = []
    if n >= 2:
        shape = rng.randint(0, 3)
        for v in range(2, n + 1):
            if shape == 0:
                # random attach (general tree)
                p = rng.randint(1, v - 1)
            elif shape == 1:
                # path-like: attach to previous -> long chains, stress depth
                p = v - 1
            elif shape == 2:
                # star-ish: attach mostly to root
                p = 1 if rng.random() < 0.7 else rng.randint(1, v - 1)
            else:
                # caterpillar / balanced-ish
                p = rng.randint(max(1, v - 3), v - 1)
            # randomly orient the edge
            a, b = (p, v) if rng.random() < 0.5 else (v, p)
            edges.append((a, b))
        rng.shuffle(edges)

    # Also randomly relabel nodes so root 1 is not special in the input.
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    relabel = {i + 1: perm[i] for i in range(n)}

    lines = [str(n)]
    for a, b in edges:
        lines.append(f"{relabel[a]} {relabel[b]}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
