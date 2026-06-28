#!/usr/bin/env python3
"""Deterministic local scorer for "Flood-Control Levee Placement".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance is an H x W grid of integer HEIGHTS h[r][c], a levee budget B,
    and S flood SOURCE cells. A SOLUTION is a list of levee cells:
        first an integer L (the number of levees placed),
        then L lines "r c" giving the (row, col) of each levee.
  * FLOOD SEMANTICS. Water starts at the source cells and spreads to a 4-adjacent
    neighbour `v` from an already-flooded cell `u` whenever h[u] >= h[v] (downhill
    or level, never strictly uphill). A leveed cell is impassable: it never floods
    and water cannot flow through it. Source cells are always flooded.
  * FEASIBILITY FLOOR. A solution is INFEASIBLE (score 0) iff any of:
      (a) the output does not parse as "L" then L valid "r c" pairs;
      (b) L > B (over budget);
      (c) any levee cell is out of the grid;
      (d) any levee coincides with a SOURCE cell;
      (e) two levees occupy the same cell (a duplicate).
    Placing FEWER than B levees (including zero) is allowed and feasible.
  * OBJECTIVE. For a feasible solution, run the flood simulation with the levees
    in place and count `flooded` = number of FLOODED cells (levee cells do not
    count as flooded -- they are dry barriers; source cells DO count as flooded).
    Fewer flooded cells is better.
  * REFERENCE. `flooded_ref` = the flooded-cell count with NO levees placed
    (the do-nothing baseline), recomputed inside the scorer so it is reproducible
    and solver-independent.
  * SCORE = round(1_000_000 * flooded_ref / flooded_solver) for a feasible
    solution with flooded_solver > 0. The no-levee baseline scores ~1_000_000;
    a solver that keeps cells dry scores strictly more; a feasible solution that
    somehow floods MORE (impossible -- levees only ever block, never help water --
    but guarded anyway) scores less but stays positive. flooded_solver == 0 is
    impossible because the source cells are always flooded and can't be leveed, so
    it never arises; we guard it with a full-credit cap regardless.

The scorer is self-contained and deterministic: it recomputes the no-levee
reference itself, so the baseline is reproducible and solver-independent.
"""
import sys
from collections import deque


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    H = int(next(it))
    W = int(next(it))
    B = int(next(it))
    S = int(next(it))
    grid = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            grid[r][c] = int(next(it))
    sources = []
    for _ in range(S):
        sr = int(next(it))
        sc = int(next(it))
        sources.append((sr, sc))
    return H, W, B, S, grid, sources


def read_solution(path):
    """Return a list of (r,c) levee cells, or None on a hard parse error.

    A solution is: an integer L, then L pairs "r c". We return the raw list of
    pairs; range/duplicate/source/budget validation happens in main().
    """
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if not toks:
        return None
    it = iter(toks)
    try:
        L = int(next(it))
    except (StopIteration, ValueError):
        return None
    if L < 0:
        return None
    levees = []
    for _ in range(L):
        try:
            r = int(next(it))
            c = int(next(it))
        except (StopIteration, ValueError):
            return None
        levees.append((r, c))
    return levees


def flood_count(H, W, grid, sources, blocked):
    """Multi-source flood. `blocked` is a set of (r,c) levee cells. Returns the
    number of flooded cells (sources count, levees never flood)."""
    flooded = [[False] * W for _ in range(H)]
    q = deque()
    for (sr, sc) in sources:
        if 0 <= sr < H and 0 <= sc < W and (sr, sc) not in blocked:
            if not flooded[sr][sc]:
                flooded[sr][sc] = True
                q.append((sr, sc))
    cnt = len(q)
    while q:
        r, c = q.popleft()
        hu = grid[r][c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if flooded[nr][nc]:
                continue
            if (nr, nc) in blocked:
                continue
            # water flows from u to v iff h[u] >= h[v] (downhill or level)
            if hu >= grid[nr][nc]:
                flooded[nr][nc] = True
                cnt += 1
                q.append((nr, nc))
    return cnt


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    H, W, B, S, grid, sources = read_instance(sys.argv[1])

    source_set = set(sources)

    # reference: no levees at all -- recomputed by the scorer, solver-independent
    flooded_ref = flood_count(H, W, grid, sources, set())

    levees = read_solution(sys.argv[2])
    if levees is None:
        print(0)  # parse error -> infeasible
        return

    if len(levees) > B:
        print(0)  # over budget -> infeasible
        return

    seen = set()
    for (r, c) in levees:
        if r < 0 or r >= H or c < 0 or c >= W:
            print(0)  # out of grid -> infeasible
            return
        if (r, c) in source_set:
            print(0)  # levee on a source -> infeasible
            return
        if (r, c) in seen:
            print(0)  # duplicate levee -> infeasible
            return
        seen.add((r, c))

    flooded_solver = flood_count(H, W, grid, sources, seen)
    if flooded_solver <= 0:
        # impossible (sources always flood and can't be leveed), but guard the
        # division: a zero-flood solution gets a generous full-credit cap.
        print(2_000_000)
        return

    score = int(round(1_000_000.0 * flooded_ref / flooded_solver))
    print(score)


if __name__ == "__main__":
    main()
