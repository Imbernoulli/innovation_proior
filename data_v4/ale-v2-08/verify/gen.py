#!/usr/bin/env python3
"""Instance generator for "Facility Layout Assignment" (the Quadratic Assignment
Problem, ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    n
    F[0][0] F[0][1] ... F[0][n-1]
    ...                                (n rows: the flow matrix)
    F[n-1][0] ... F[n-1][n-1]
    D[0][0] D[0][1] ... D[0][n-1]
    ...                                (n rows: the distance matrix)
    D[n-1][0] ... D[n-1][n-1]

with non-negative integer entries. The size n is chosen deterministically from
the seed in [60, 120]. The flow matrix is a symmetric, hollow (zero diagonal)
matrix that mixes a few high-traffic "communicating groups" of facilities (a
block structure) with a sparse light background, so a good assignment must place
each group's heavy-flow members onto mutually nearby locations. The distance
matrix is the Euclidean (rounded) distance between n points placed on a grid
(itself a metric), which is the regime where a clever placement beats the
identity by a wide margin and where local search has the most slack to recover.

The block-structured flow + metric distance combination is the classic hard
regime for QAP: the identity assignment ignores the group structure entirely, so
there is a large gap for a strong heuristic to close.
"""
import sys
import random

GRID = 100  # locations live on a [0,GRID]^2 integer grid


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x5EED_0008 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # problem size: deterministic from the seed, in [60, 120]
    n = rng.randint(60, 120)

    # ---- distance matrix D: rounded Euclidean distance between n grid points ----
    pts = []
    seen = set()
    while len(pts) < n:
        x = rng.randint(0, GRID)
        y = rng.randint(0, GRID)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        pts.append((x, y))
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = pts[i]
        for j in range(n):
            xj, yj = pts[j]
            D[i][j] = int(round(((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5))

    # ---- flow matrix F: block "communicating groups" + sparse light background ----
    # partition facilities into a handful of groups; heavy flow inside a group,
    # light random flow everywhere else. Symmetric, zero diagonal.
    F = [[0] * n for _ in range(n)]
    num_groups = rng.randint(4, 8)
    # random group assignment
    group = [rng.randrange(num_groups) for _ in range(n)]
    heavy_lo, heavy_hi = 20, 100   # intra-group (heavy) flow range
    light_p = 0.15                 # probability of a light background edge
    light_lo, light_hi = 1, 8      # light background flow range
    for i in range(n):
        for j in range(i + 1, n):
            if group[i] == group[j] and rng.random() < 0.7:
                w = rng.randint(heavy_lo, heavy_hi)
            elif rng.random() < light_p:
                w = rng.randint(light_lo, light_hi)
            else:
                w = 0
            F[i][j] = w
            F[j][i] = w

    out = [str(n)]
    for i in range(n):
        out.append(" ".join(str(F[i][j]) for j in range(n)))
    for i in range(n):
        out.append(" ".join(str(D[i][j]) for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
