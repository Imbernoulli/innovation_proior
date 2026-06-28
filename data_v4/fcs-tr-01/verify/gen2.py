#!/usr/bin/env python3
"""Adversarial generator: deep chains, stars, caterpillars, broom shapes.
Usage: gen2.py <seed> [maxn] [maxq]"""
import sys, random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    maxq = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rng = random.Random(seed * 7919 + 13)
    n = rng.randint(1, maxn)
    shape = rng.choice(["chain", "star", "caterpillar", "broom", "random_shallow"])
    edges = []
    if shape == "chain":
        for v in range(2, n + 1):
            edges.append((v - 1, v))
    elif shape == "star":
        for v in range(2, n + 1):
            edges.append((1, v))
    elif shape == "caterpillar":
        spine = max(1, n // 2)
        for v in range(2, spine + 1):
            edges.append((v - 1, v))
        for v in range(spine + 1, n + 1):
            edges.append((rng.randint(1, spine), v))
    elif shape == "broom":
        h = max(1, n // 2)
        for v in range(2, h + 1):
            edges.append((v - 1, v))
        for v in range(h + 1, n + 1):
            edges.append((h, v))
    else:  # random_shallow: small parent gap -> shallow
        for v in range(2, n + 1):
            edges.append((max(1, v - rng.randint(1, 3)), v))

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
