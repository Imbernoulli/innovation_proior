#!/usr/bin/env python3
"""
Random small-case generator for fcs-gr-06.

Usage: python3 gen.py <seed>
Emits a valid grid instance to stdout:
    R C K
    R lines of the grid (chars in {'.', '#', 'S', 'T'})

Exactly one 'S' and one 'T'. Wall density and K are randomized so that both
reachable and unreachable cases (and 0/1/multi-break optimal paths) appear.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    R = rng.randint(1, 6)
    C = rng.randint(1, 6)
    # ensure at least 2 cells so S and T can be distinct
    while R * C < 2:
        R = rng.randint(1, 6)
        C = rng.randint(1, 6)

    K = rng.randint(0, 4)
    wall_p = rng.choice([0.0, 0.15, 0.3, 0.45, 0.6, 0.75])

    grid = [['.' for _ in range(C)] for _ in range(R)]
    for i in range(R):
        for j in range(C):
            if rng.random() < wall_p:
                grid[i][j] = '#'

    # pick two distinct cells for S and T (overwrite whatever was there)
    cells = [(i, j) for i in range(R) for j in range(C)]
    rng.shuffle(cells)
    (sr, sc), (tr, tc) = cells[0], cells[1]
    grid[sr][sc] = 'S'
    grid[tr][tc] = 'T'

    out = [f"{R} {C} {K}"]
    for i in range(R):
        out.append("".join(grid[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
