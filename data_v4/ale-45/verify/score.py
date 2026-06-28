#!/usr/bin/env python3
"""Deterministic local scorer for ale-45 "Watchman Route (max area seen, bounded steps)".

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE
prints a single integer: the score (number of distinct cells ever seen along the
route), or 0 if the solution is infeasible.

Solution format (stdout of the solver):
    one token: the MOVE STRING -- a (possibly empty) sequence of characters over
    the alphabet {U, D, L, R}.  Starting at 'S', each character moves the guard
    one cell:  U = row-1, D = row+1, L = col-1, R = col+1.  The empty route
    (no moves) is allowed and stays at the start.

    To accept an empty route robustly, the solver may instead print a single
    token "0" or "-" to denote "no moves"; both are treated as the empty route.

Feasibility -> 0 floor.  The output is INFEASIBLE (score forced to 0) if:
  * it is missing / has more than one token;
  * the move string contains a character other than U,D,L,R (after the
    empty-route sentinels "0"/"-");
  * its length exceeds the budget L (over budget);
  * any move steps off the grid or onto a wall '#' (walks into a wall);
  * the route is OPEN, i.e. it does not return to the start cell at the end.

Score (when feasible): replay the route; the set of VISITED cells is the start
plus every cell stepped onto.  Visibility from a cell is ROOK line-of-sight: the
cell sees itself and every free cell reachable in a straight horizontal or
vertical run not blocked by a wall (each run stops at the first wall).  Score =
size of the union of the visibility sets of all visited cells = number of
distinct cells ever seen.
"""
import sys

# move char -> (dr, dc)
MOVE = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


def read_grid(path):
    with open(path) as f:
        lines = f.read().split('\n')
    # first non-empty line: H W L
    i = 0
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    H, W, L = map(int, lines[i].split())
    i += 1
    grid = []
    for r in range(H):
        row = lines[i + r]
        # pad/truncate defensively to width W
        if len(row) < W:
            row = row + '.' * (W - len(row))
        grid.append(row[:W])
    # locate start
    sr = sc = -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                sr, sc = r, c
    return H, W, L, grid, sr, sc


def visibility_set(H, W, grid, r, c):
    """Rook line-of-sight cells visible from (r,c).  Returns a set of (r,c)."""
    seen = set()
    seen.add((r, c))                       # sees itself
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        while 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
            seen.add((nr, nc))
            nr += dr
            nc += dc
    return seen


def main():
    H, W, L, grid, sr, sc = read_grid(sys.argv[1])

    # The start must be a free cell (it is, by construction: 'S').
    if sr < 0:
        print(0); return

    # Parse the solution as whitespace tokens.
    try:
        with open(sys.argv[2]) as f:
            tokens = f.read().split()
    except OSError:
        print(0); return

    if len(tokens) == 0:
        # No output at all -> infeasible.
        print(0); return
    if len(tokens) > 1:
        # More than one token -> malformed.
        print(0); return

    move_str = tokens[0]
    if move_str in ('0', '-'):
        move_str = ''                      # empty-route sentinel

    # Validate alphabet.
    for ch in move_str:
        if ch not in MOVE:
            print(0); return

    # Budget check.
    if len(move_str) > L:
        print(0); return

    # Replay route, collecting visited cells; reject wall/off-grid moves.
    r, c = sr, sc
    visited = [(sr, sc)]
    for ch in move_str:
        dr, dc = MOVE[ch]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            print(0); return              # off grid
        if grid[nr][nc] == '#':
            print(0); return              # walked into a wall
        r, c = nr, nc
        visited.append((r, c))

    # Closed-route check: must return to start.
    if (r, c) != (sr, sc):
        print(0); return

    # Union the rook-visibility of every visited cell.
    seen = set()
    # cache per-cell visibility (a route revisits cells a lot)
    cache = {}
    for (vr, vc) in visited:
        key = (vr, vc)
        vis = cache.get(key)
        if vis is None:
            vis = visibility_set(H, W, grid, vr, vc)
            cache[key] = vis
        seen |= vis

    print(len(seen))


if __name__ == "__main__":
    main()
