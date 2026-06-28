#!/usr/bin/env python3
"""Deterministic local scorer for ale-29 "Connected Region Selection".

Usage: python3 score.py INSTANCE_FILE SOLUTION_FILE
Prints a single float: the score.

Scoring rule (see context.md):
  The solution lists K cells. It is FEASIBLE iff
    (1) 0 <= K <= B,
    (2) every listed cell (r, c) is in range 0<=r<H, 0<=c<W,
    (3) the listed cells are pairwise distinct,
    (4) the listed cells form a SINGLE 4-connected component
        (K == 0 is allowed -- the empty region -- and is trivially feasible).
  If feasible, the raw score is the SUM of the weights of the chosen cells.
  If INFEASIBLE (out of range, duplicate, over budget, or more than one
  connected component), the score is floored to 0.0.

  Because the empty region is always feasible with sum 0, a correct solver can
  always guarantee score >= 0; a positive score requires actually enclosing
  net-positive weight while staying connected and within budget.

Note: the raw objective can in principle be negative if a solver chose a
net-negative connected set, but the scorer never returns below 0 -- a feasible
but net-negative selection is dominated by the empty region, so we report
max(0, sum) for feasible outputs as well (the empty region is always available).
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); B = int(next(it))
    grid = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            grid[r][c] = int(next(it))
    return H, W, B, grid


def read_solution(path):
    """Solution format:
        line 1: K   (number of chosen cells)
        next K lines: r c
    Returns list of (r, c) or raises on a parse error.
    """
    with open(path) as f:
        toks = f.read().split()
    if not toks:
        # treat an empty file as the empty region
        return []
    it = iter(toks)
    K = int(next(it))
    cells = []
    for _ in range(K):
        r = int(next(it)); c = int(next(it))
        cells.append((r, c))
    return cells


def is_feasible(cells, H, W, B):
    K = len(cells)
    if K > B:
        return False, "over budget"
    if K == 0:
        return True, "empty"
    seen = set()
    for (r, c) in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False, f"out of range {(r, c)}"
        if (r, c) in seen:
            return False, f"duplicate {(r, c)}"
        seen.add((r, c))
    # single 4-connected component via BFS
    start = cells[0]
    stack = [start]
    visited = {start}
    while stack:
        (r, c) = stack.pop()
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in seen and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    if len(visited) != K:
        return False, "not 4-connected"
    return True, "ok"


def score(instance_path, solution_path):
    H, W, B, grid = read_instance(instance_path)
    try:
        cells = read_solution(solution_path)
    except (StopIteration, ValueError):
        return 0.0
    ok, _why = is_feasible(cells, H, W, B)
    if not ok:
        return 0.0
    total = 0
    for (r, c) in cells:
        total += grid[r][c]
    # empty region (sum 0) is always available, so feasible score never < 0
    return float(max(0, total))


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION\n")
        sys.exit(1)
    s = score(sys.argv[1], sys.argv[2])
    print(s)


if __name__ == "__main__":
    main()
