#!/usr/bin/env python3
"""Deterministic local scorer for "Grid Light Placement".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance is an H x W grid of '.' (floor cells, which MUST be lit) and '#'
    (walls, which block light and cannot hold a light).
  * A SOLUTION is a count K followed by K light positions "r c" (0-indexed).
    A light at (r, c) illuminates its own cell and every cell reachable from it
    along the same row or the same column WITHOUT crossing a wall (it stops just
    before the first '#' in each of the four directions, or the grid edge).
  * FEASIBILITY -> score 0 if ANY of these fail:
      - the file does not parse as "K" then K integer pairs in range,
      - a light is placed on a wall cell ('#') or outside the grid,
      - after simulating illumination from every light, some floor cell '.'
        remains dark (unlit).
  * Otherwise let K = the number of lights used. Let B = the number of maximal
    horizontal floor corridors in the grid (the "one-light-per-horizontal-
    corridor" reference, which is always feasible: putting one light anywhere in
    each horizontal corridor lights every floor cell). The score is
        score = round(1_000_000 * B / K)        (feasible, K >= 1)
    Fewer lights -> higher score; the reference scores exactly 1_000_000.
    A grid with zero floor cells is a degenerate full-credit case (score
    1_000_000) and a feasible solution there uses K = 0 lights.

The scorer is self-contained and deterministic: it does not trust the solver and
re-simulates illumination and recomputes B itself.
"""
import sys


def read_instance(path):
    with open(path) as f:
        lines = f.read().split("\n")
    # first non-empty line is "H W"
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    H, W = map(int, lines[idx].split())
    idx += 1
    grid = []
    for r in range(H):
        row = lines[idx + r]
        # pad / truncate defensively to width W
        if len(row) < W:
            row = row + "." * (W - len(row))
        grid.append(row[:W])
    return H, W, grid


def count_h_corridors(H, W, grid):
    """Number of maximal horizontal runs of floor cells (the reference B)."""
    B = 0
    for r in range(H):
        in_run = False
        for c in range(W):
            if grid[r][c] == '.':
                if not in_run:
                    B += 1
                    in_run = True
            else:
                in_run = False
    return B


def read_solution(path, H, W, grid):
    """Return list of (r, c) light positions if well-formed and on floor cells,
    else None."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if not toks:
        return None
    it = iter(toks)
    try:
        K = int(next(it))
    except (StopIteration, ValueError):
        return None
    if K < 0:
        return None
    lights = []
    for _ in range(K):
        try:
            r = int(next(it))
            c = int(next(it))
        except (StopIteration, ValueError):
            return None
        if r < 0 or r >= H or c < 0 or c >= W:
            return None
        if grid[r][c] != '.':
            return None  # light on a wall -> infeasible
        lights.append((r, c))
    # any trailing junk tokens -> reject
    extra = list(it)
    if extra:
        return None
    return lights


def simulate(H, W, grid, lights):
    """Return a 2-D boolean lit map after casting light from every source."""
    lit = [[False] * W for _ in range(H)]
    for (r, c) in lights:
        lit[r][c] = True
        # right
        cc = c + 1
        while cc < W and grid[r][cc] == '.':
            lit[r][cc] = True
            cc += 1
        # left
        cc = c - 1
        while cc >= 0 and grid[r][cc] == '.':
            lit[r][cc] = True
            cc -= 1
        # down
        rr = r + 1
        while rr < H and grid[rr][c] == '.':
            lit[rr][c] = True
            rr += 1
        # up
        rr = r - 1
        while rr >= 0 and grid[rr][c] == '.':
            lit[rr][c] = True
            rr -= 1
    return lit


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    H, W, grid = read_instance(sys.argv[1])

    num_floor = sum(row.count('.') for row in grid)
    B = count_h_corridors(H, W, grid)

    lights = read_solution(sys.argv[2], H, W, grid)
    if lights is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    if num_floor == 0:
        # No floor cells to light: feasible iff no lights claimed needed.
        # Any K == 0 is full credit; K > 0 cannot be placed anyway (no floor).
        print(1_000_000)
        return

    lit = simulate(H, W, grid, lights)
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '.' and not lit[r][c]:
                print(0)  # a floor cell stayed dark -> INFEASIBLE
                return

    K = len(lights)
    if K <= 0:
        # floor cells exist but no lights -> they cannot all be lit; defensive.
        print(0)
        return

    score = int(round(1_000_000.0 * B / K))
    print(score)


if __name__ == "__main__":
    main()
