#!/usr/bin/env python3
"""Deterministic local scorer for "Maze Treasure Collection (time-budgeted)".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance is an H x W maze with walls '#', floor '.', a unique start 'S',
    a step budget T, and K treasures (each a value v on an open cell).
  * A SOLUTION is a move string over {'U','D','L','R'} (whitespace ignored).
    The agent starts at S; each character moves one cell up/down/left/right.
  * FEASIBILITY (any violation => score 0):
      - the move string has length > T (too many steps), OR
      - any move steps off the grid or onto a wall '#'.
    An empty move string is feasible (collects whatever is on S, i.e. nothing,
    since S itself never holds a treasure).
  * Collected value = sum of v over the DISTINCT treasure cells the agent ever
    occupies along the walk (the start cell counts; each treasure counts once
    even if revisited).
  * Normalization: let B = the collected value of the scorer's own deterministic
    greedy baseline (repeatedly BFS to the nearest reachable uncollected treasure
    and walk there while the remaining budget allows). B is recomputed inside the
    scorer, so the reference is reproducible and independent of the solver.
  * SCORE = round(1_000_000 * collected / B) for a feasible walk with B > 0.
    If B == 0 (no treasure reachable at all within T -- should not happen for
    generated instances) then SCORE = 1_000_000 for any feasible walk, since the
    baseline collects nothing. INFEASIBLE -> 0.

The scorer is self-contained and deterministic: it does not trust the solver and
recomputes B itself.
"""
import sys
from collections import deque

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def read_instance(path):
    with open(path) as f:
        lines = f.read().split("\n")
    # first non-empty line: H W T
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    H, W, T = map(int, lines[idx].split())
    idx += 1
    grid = []
    for _ in range(H):
        # grid rows are taken verbatim (they may contain only '#','.','S')
        row = lines[idx]
        idx += 1
        # pad / trim defensively to width W
        if len(row) < W:
            row = row + "." * (W - len(row))
        grid.append(row[:W])
    # K
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    K = int(lines[idx].split()[0])
    idx += 1
    treasures = {}
    for _ in range(K):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        r, c, v = map(int, lines[idx].split())
        idx += 1
        treasures[(r, c)] = v
    # locate start
    sr = sc = -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == "S":
                sr, sc = r, c
    return H, W, T, grid, treasures, sr, sc


def is_wall(grid, H, W, r, c):
    if r < 0 or r >= H or c < 0 or c >= W:
        return True
    return grid[r][c] == "#"


def read_solution(path):
    """Return the move string (only the U/D/L/R characters), or None on error."""
    try:
        with open(path) as f:
            raw = f.read()
    except OSError:
        return None
    moves = []
    for ch in raw:
        if ch in DIRS:
            moves.append(ch)
        elif ch.isspace():
            continue
        else:
            # any stray non-move, non-space token => malformed
            return None
    return "".join(moves)


def replay_value(grid, H, W, T, treasures, sr, sc, moves):
    """Replay the walk. Return collected value, or None if infeasible."""
    if len(moves) > T:
        return None
    r, c = sr, sc
    collected = 0
    seen = set()
    # the start cell never holds a treasure (generator guarantees), but be safe:
    if (r, c) in treasures and (r, c) not in seen:
        seen.add((r, c))
        collected += treasures[(r, c)]
    for ch in moves:
        dr, dc = DIRS[ch]
        nr, nc = r + dr, c + dc
        if is_wall(grid, H, W, nr, nc):
            return None  # stepped off-grid or into a wall
        r, c = nr, nc
        if (r, c) in treasures and (r, c) not in seen:
            seen.add((r, c))
            collected += treasures[(r, c)]
    return collected


def greedy_baseline(grid, H, W, T, treasures, sr, sc):
    """Deterministic greedy: repeatedly walk to the nearest uncollected treasure
    (BFS shortest path, ties broken by (r,c)) while the budget allows. Returns
    the collected value. This is the reproducible reference B."""
    remaining = T
    r, c = sr, sc
    collected = 0
    got = set()
    targets = set(treasures.keys())
    while True:
        # BFS from (r,c) to find the nearest uncollected treasure within remaining
        dist = [[-1] * W for _ in range(H)]
        dist[r][c] = 0
        dq = deque([(r, c)])
        best = None  # (dist, r, c)
        while dq:
            cr, cc = dq.popleft()
            d = dist[cr][cc]
            if d > remaining:
                continue
            if (cr, cc) in targets and (cr, cc) not in got:
                cand = (d, cr, cc)
                if best is None or cand < best:
                    best = cand
                    # do not break: we want the globally nearest, but BFS layers
                    # are monotone so the first-found at minimal d with min (r,c)
                    # is reached by continuing the same layer. We keep scanning
                    # the current and not deeper layers.
            for ch in DIRS:
                dr, dc = DIRS[ch]
                nr, nc = cr + dr, cc + dc
                if is_wall(grid, H, W, nr, nc):
                    continue
                if dist[nr][nc] != -1:
                    continue
                if d + 1 > remaining:
                    continue
                dist[nr][nc] = d + 1
                dq.append((nr, nc))
        if best is None:
            break
        bd, br, bc = best
        remaining -= bd
        r, c = br, bc
        got.add((br, bc))
        collected += treasures[(br, bc)]
    return collected


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    H, W, T, grid, treasures, sr, sc = read_instance(sys.argv[1])

    moves = read_solution(sys.argv[2])
    if moves is None:
        print(0)  # malformed token => INFEASIBLE
        return

    collected = replay_value(grid, H, W, T, treasures, sr, sc, moves)
    if collected is None:
        print(0)  # too many steps or walked into a wall => INFEASIBLE
        return

    B = greedy_baseline(grid, H, W, T, treasures, sr, sc)
    if B <= 0:
        # baseline collected nothing reachable; any feasible walk gets full credit
        print(1_000_000)
        return

    score = int(round(1_000_000.0 * collected / B))
    print(score)


if __name__ == "__main__":
    main()
