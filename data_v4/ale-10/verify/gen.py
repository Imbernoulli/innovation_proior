#!/usr/bin/env python3
"""Instance generator for Wall Painting (ALE-10).

Usage: python3 gen.py SEED  > instance.txt

A target picture is an N x N grid of colours in 0..C-1. The solver paints the
picture by emitting at most T axis-aligned rectangular *brush strokes*; each
stroke paints a solid rectangle with one colour onto a canvas that starts filled
with colour 0, and later strokes overwrite earlier ones. The score is the number
of canvas cells whose final colour equals the target. The pictures here are built
from a handful of overlapping coloured rectangles plus a little per-cell noise,
so they are *almost* reconstructible by a small number of strokes but never
exactly (the noise and the overlaps create an irreducible optimisation gap), and
T is too small to paint every cell individually -> a genuine heuristic problem.

Instance format (stdin of the solver):
    N C T
    N lines, each with N integers in 0..C-1   (the target grid; row-major)
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed * 1_000_003 + 777)

    # Grid side, palette size, operation budget.
    N = rng.randint(20, 40)
    C = rng.randint(3, 6)
    # Budget: enough strokes to do real layering, but far fewer than N*N cells,
    # so painting cell-by-cell is hopeless and structure must be exploited.
    T = rng.randint(N, 3 * N)

    # Background colour 0 everywhere (matches the canvas's initial colour).
    grid = [[0] * N for _ in range(N)]

    # Lay down R overlapping coloured rectangles (the "true" picture structure).
    R = rng.randint(6, 18)
    for _ in range(R):
        r1 = rng.randint(0, N - 1)
        r2 = rng.randint(0, N - 1)
        c1 = rng.randint(0, N - 1)
        c2 = rng.randint(0, N - 1)
        if r1 > r2:
            r1, r2 = r2, r1
        if c1 > c2:
            c1, c2 = c2, c1
        col = rng.randint(0, C - 1)
        for r in range(r1, r2 + 1):
            row = grid[r]
            for c in range(c1, c2 + 1):
                row[c] = col

    # Per-cell salt-and-pepper noise: makes exact reconstruction impossible and
    # forces the solver to trade coverage of clean regions against noisy ones.
    noise = 0.05 + 0.05 * rng.random()  # 5%..10% of cells flipped
    for r in range(N):
        for c in range(N):
            if rng.random() < noise:
                grid[r][c] = rng.randint(0, C - 1)

    out = [f"{N} {C} {T}"]
    for r in range(N):
        out.append(" ".join(str(x) for x in grid[r]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
