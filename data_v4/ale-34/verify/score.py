#!/usr/bin/env python3
"""Deterministic scorer for ale-34 "Maze Carving to a Target Difficulty".

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE
prints a single number: the score (>=0). 0 means infeasible.

Scoring rule (must match context.md exactly):
  - The solution lists exactly B cells "r c" to carve open.
  - Feasibility (else score = 0, the floor):
        * exactly B cells are listed;
        * every cell is in bounds;
        * every cell is currently a WALL ('#') in the instance grid;
        * the B cells are pairwise distinct;
        * start and end are NOT carved-over-walls (they are already open; listing
          them would mean they were walls, which they are not -> rejected above);
        * after carving the B cells open, the end is reachable from the start
          by 4-adjacent moves over open cells.
  - Score = the BFS shortest-path length (number of edges / steps) from start to
    end over the carved grid. By construction this is >= 1 (S != T). If the end
    is unreachable, the score is 0.
"""
import sys
from collections import deque


def read_instance(path):
    with open(path) as f:
        toks = f.read().split('\n')
    # First line: H W B
    idx = 0
    while toks[idx].strip() == '':
        idx += 1
    H, W, B = map(int, toks[idx].split())
    idx += 1
    while toks[idx].strip() == '':
        idx += 1
    sr, sc, tr, tc = map(int, toks[idx].split())
    idx += 1
    grid = []
    r = 0
    while r < H:
        line = toks[idx]
        idx += 1
        if line == '' and len(line) == 0:
            # blank line that is not a grid row only if grid not full width;
            # grid rows have exactly W chars, never empty here.
            if len(grid) < H and W == 0:
                pass
            continue
        grid.append(list(line[:W]))
        r += 1
    return H, W, B, sr, sc, tr, tc, grid


def read_solution(path, B):
    with open(path) as f:
        nums = f.read().split()
    try:
        vals = list(map(int, nums))
    except ValueError:
        return None
    # Expect exactly 2*B integers (B pairs). Trailing/leading whitespace ignored.
    cells = []
    if len(vals) != 2 * B:
        return None
    for i in range(B):
        cells.append((vals[2 * i], vals[2 * i + 1]))
    return cells


def score(instance_path, solution_path):
    H, W, B, sr, sc, tr, tc, grid = read_instance(instance_path)
    cells = read_solution(solution_path, B)
    if cells is None:
        return 0  # wrong number of carved cells

    seen = set()
    for (r, c) in cells:
        if not (0 <= r < H and 0 <= c < W):
            return 0  # out of bounds
        if grid[r][c] != '#':
            return 0  # not currently a wall (already open / S / T)
        if (r, c) in seen:
            return 0  # duplicate
        seen.add((r, c))

    # Carve.
    for (r, c) in cells:
        grid[r][c] = '.'

    if grid[sr][sc] != '.' or grid[tr][tc] != '.':
        return 0  # start/end blocked (should not happen by construction)

    # BFS shortest path start -> end over open cells.
    dist = [[-1] * W for _ in range(H)]
    dq = deque()
    dist[sr][sc] = 0
    dq.append((sr, sc))
    while dq:
        r, c = dq.popleft()
        if r == tr and c == tc:
            return dist[r][c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.' and dist[nr][nc] < 0:
                dist[nr][nc] = dist[r][c] + 1
                dq.append((nr, nc))
    return 0  # end unreachable -> infeasible floor


if __name__ == '__main__':
    print(score(sys.argv[1], sys.argv[2]))
