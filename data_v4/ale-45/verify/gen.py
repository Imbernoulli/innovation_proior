#!/usr/bin/env python3
"""Instance generator for ale-45 "Watchman Route (max area seen, bounded steps)".

Usage: python3 gen.py SEED  ->  writes one instance to stdout.

Instance format (stdin of the solver):
    H W L               grid dimensions and the step budget L
    H lines of W chars  the grid: '.' = free cell, '#' = wall, 'S' = start
                        (exactly one 'S'; the start cell is also free/walkable)

The guard occupies a free cell and may move 4-directionally (up/down/left/right)
to an adjacent free cell; each move is one "step".  A route is a sequence of
cells starting and ending at 'S'.  Visibility from a cell is ROOK line-of-sight:
the cell sees itself and every free cell reachable in a straight horizontal or
vertical run that is not blocked by a wall (the run stops at the first wall).

The grid is connected over its free cells (guaranteed by construction), and the
walls form interior "rooms/pillars" so that line-of-sight (not just reachability)
is the binding constraint -- a sweep that only walks a corridor misses cells in
side pockets it never looks down.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(seed * 1_000_003 + 777)

    # Grid size grows mildly with seed so the seed set spans small..medium.
    H = rng.randint(28, 34)
    W = rng.randint(28, 34)

    # Start with all free cells.
    grid = [['.' for _ in range(W)] for _ in range(H)]

    # Carve interior walls: a mixture of short bars and pillars.  We keep the
    # border free (so the guard can always circulate) and ensure connectivity of
    # the free cells afterwards by rejecting walls that would disconnect.
    cells = H * W
    target_walls = int(cells * rng.uniform(0.16, 0.22))

    def free_count_and_connected():
        # BFS over free cells from the first free cell; return (count, connected?)
        start = None
        for r in range(H):
            for c in range(W):
                if grid[r][c] != '#':
                    start = (r, c)
                    break
            if start:
                break
        seen = set([start])
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        total_free = sum(1 for r in range(H) for c in range(W) if grid[r][c] != '#')
        return total_free, len(seen) == total_free

    placed = 0
    attempts = 0
    while placed < target_walls and attempts < target_walls * 30:
        attempts += 1
        # Bias toward the interior so the border stays open.
        r = rng.randint(1, H - 2)
        c = rng.randint(1, W - 2)
        if grid[r][c] == '#':
            continue
        grid[r][c] = '#'
        _, conn = free_count_and_connected()
        if not conn:
            grid[r][c] = '.'        # would disconnect -> revert
            continue
        placed += 1

    # Choose the start cell among free cells (prefer near a corner so a naive
    # sweep is the obvious baseline but not optimal).
    free_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != '#']
    # Pick a free cell closest to the top-left corner.
    free_cells.sort(key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    sr, sc = free_cells[0]
    grid[sr][sc] = 'S'

    # Step budget L: a TIGHT fraction of the free-cell count.  A closed route of
    # this length cannot stand near enough cells to see them all (rook
    # line-of-sight still leaves side pockets behind walls unseen), so the route
    # must SELECT which pockets to inspect -- this is what makes the problem a
    # genuine budgeted orienteering / coverage problem rather than a trivial
    # "walk everywhere" sweep.  Empirically L ~ 0.25..0.35 * freeN keeps the best
    # achievable coverage well below the ceiling.
    free_n = len(free_cells)
    L = int(free_n * rng.uniform(0.25, 0.35))
    # Floor so even tiny instances admit a non-trivial closed loop.
    L = max(L, 2 * (H + W))

    out = [f"{H} {W} {L}"]
    for r in range(H):
        out.append("".join(grid[r]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
