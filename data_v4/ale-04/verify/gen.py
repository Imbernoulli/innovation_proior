#!/usr/bin/env python3
"""Instance generator for "Heat-Diffusion Tile Coloring" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    N W
    h_0_0 t_0_0 p_0_0   h_0_1 t_0_1 p_0_1 ...   (row 0: N triples)
    ...                                          (N rows total)

Semantics (see context.md for the full statement):

  * The board is an N x N grid of tiles; each tile gets a binary coating x in {0,1}.
  * W (>= 1) is the global *interface* weight: every pair of 4-adjacent tiles with
    DIFFERENT coatings costs W (a smoothness / heat-roughness penalty).
  * For each cell, a triple (h, t, p):
      - h >= 0  : the *field strength* (how strongly this cell prefers its target).
      - t in {0,1}: the *target* coating the cell would like to take.
      - p in {-1,0,1}: pin flag. p == -1 means free; p in {0,1} means the cell is
        PINNED to coating p (the solver MUST output that value there).
    The unary cost of cell i is  h_i * [x_i != t_i]   (0 if it matches its target).

The targets t form a few spatial blobs, so honoring every cell's target produces a
ragged, high-interface coloring while a perfectly smooth coloring betrays many cells:
the optimum lives on the interface, which is exactly where the relaxation + boundary
local search has to work. Pins anchor a sparse set of cells so the trivial all-zero /
all-one colorings are infeasible whenever a pin of the other color exists.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x4EA7_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # Board side and global interface weight.
    N = rng.randint(30, 60)
    W = rng.randint(3, 12)

    # Build target blobs: a handful of Gaussian "warm shoals" over a cool background.
    num_blobs = rng.randint(2, 6)
    centers = [(rng.uniform(0, N - 1), rng.uniform(0, N - 1)) for _ in range(num_blobs)]
    radii = [rng.uniform(0.10, 0.28) * N for _ in range(num_blobs)]

    target = [[0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            warm = False
            for (cy, cx), rad in zip(centers, radii):
                d2 = (r - cy) ** 2 + (c - cx) ** 2
                # soft membership: inside ~radius with a little noise on the rim
                if d2 <= (rad * rad) * rng.uniform(0.85, 1.15):
                    warm = True
                    break
            target[r][c] = 1 if warm else 0

    # Field strengths: heterogeneous, in [1, hmax]. A fraction of cells are
    # "don't care" (h == 0) so smoothness can override them for free.
    hmax = rng.randint(4, 20)
    dontcare_frac = rng.uniform(0.05, 0.25)
    field = [[0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            if rng.random() < dontcare_frac:
                field[r][c] = 0
            else:
                field[r][c] = rng.randint(1, hmax)

    # Pins: a sparse set of cells fixed to a coating. Pin a cell to its target with
    # high probability, occasionally to the opposite color to create real tension.
    pin_frac = rng.uniform(0.02, 0.08)
    pin = [[-1] * N for _ in range(N)]
    npins = 0
    for r in range(N):
        for c in range(N):
            if rng.random() < pin_frac:
                if rng.random() < 0.85:
                    pin[r][c] = target[r][c]
                else:
                    pin[r][c] = 1 - target[r][c]
                npins += 1
    # Guarantee at least one pin of each color so all-0 / all-1 are infeasible and
    # the instance is never degenerate.
    if npins == 0 or all(pin[r][c] != 0 for r in range(N) for c in range(N) if pin[r][c] != -1):
        pin[0][0] = 0
    if all(pin[r][c] != 1 for r in range(N) for c in range(N) if pin[r][c] != -1):
        pin[N - 1][N - 1] = 1

    out = []
    out.append(f"{N} {W}")
    for r in range(N):
        toks = []
        for c in range(N):
            toks.append(f"{field[r][c]} {target[r][c]} {pin[r][c]}")
        out.append(" ".join(toks))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
