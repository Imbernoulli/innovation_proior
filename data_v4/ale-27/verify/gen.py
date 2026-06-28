#!/usr/bin/env python3
"""Instance generator for "Grid Light Placement" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    H W
    row_0           (a string of length W over '.' and '#')
    row_1
    ...
    row_{H-1}

'.' is a floor cell that MUST end up lit; '#' is a wall (blocks light and cannot
hold a light). A light placed on a floor cell illuminates its own cell and extends
along its row and column in all four directions, stopping just before the first
wall (or the grid edge). The task is to light every floor cell with as few lights
as possible.

Instances are generated deterministically from the seed. The grid is a mixture of:
  * a sparse random scatter of single-cell walls (pillars), and
  * a few axis-aligned wall "bars" (partial interior walls) that chop the floor
    into many maximal horizontal/vertical corridors,
so that the corridor decomposition is non-trivial and the H-corridor / V-corridor
choice genuinely matters. We guarantee at least one floor cell exists and that the
border is open enough to keep long corridors around.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0xA11E_0027 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # Grid size: deterministic from the seed, in a band that is large enough for
    # the structure to matter but small enough to score in Python.
    H = rng.randint(30, 50)
    W = rng.randint(30, 50)

    grid = [['.' for _ in range(W)] for _ in range(H)]

    # --- single-cell wall scatter (pillars) -------------------------------
    pillar_frac = rng.uniform(0.04, 0.12)
    for r in range(H):
        for c in range(W):
            if rng.random() < pillar_frac:
                grid[r][c] = '#'

    # --- partial interior "bars" -----------------------------------------
    # Each bar is a horizontal or vertical run of walls of random length that
    # does NOT span the whole side (leaving gaps), which is what creates the
    # interesting corridor structure (a corridor on one side, another on the
    # other, joined only through the gap).
    num_bars = rng.randint(4, 10)
    for _ in range(num_bars):
        if rng.random() < 0.5:
            # horizontal bar
            r = rng.randint(1, H - 2)
            length = rng.randint(W // 4, max(W // 4, (3 * W) // 4))
            c0 = rng.randint(0, max(0, W - length))
            for c in range(c0, min(W, c0 + length)):
                grid[r][c] = '#'
        else:
            # vertical bar
            c = rng.randint(1, W - 2)
            length = rng.randint(H // 4, max(H // 4, (3 * H) // 4))
            r0 = rng.randint(0, max(0, H - length))
            for r in range(r0, min(H, r0 + length)):
                grid[r][c] = '#'

    # --- guarantee at least one floor cell --------------------------------
    floor_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == '.']
    if not floor_cells:
        r = rng.randint(0, H - 1)
        c = rng.randint(0, W - 1)
        grid[r][c] = '.'

    out = [f"{H} {W}"]
    out.extend("".join(grid[r]) for r in range(H))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
