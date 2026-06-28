#!/usr/bin/env python3
"""Deterministic local scorer for Grid Polyomino Packing (ALE-02).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE

Score = number of grid cells covered by the placed pieces, i.e. the sum of the
areas of all placements (placements are non-overlapping, so covered == area sum).
This is exactly "the scorer recomputes covered cells" from the contract.

FEASIBILITY -> 0 FLOOR. The score is 0 (the worst possible) if the solution is
infeasible for ANY of these reasons:
  * malformed / unparseable output;
  * a placement names a non-existent piece type or an out-of-range rotation;
  * a placement falls partly or wholly outside the HxW grid;
  * two placed cells coincide (overlap);
  * more than cnt_k copies of some type k are placed.
An empty placement list (P = 0) is feasible and scores 0 cells. A solver that
prints nothing therefore floors to 0 just like an infeasible one.

The script prints a single integer (the score) to stdout.
"""
import sys


def rotate(cells, rot):
    """Rotate a list of (r,c) offsets by rot*90 deg CW, then re-normalise so
    min row = min col = 0. Deterministic and matches the solver."""
    pts = cells
    for _ in range(rot % 4):
        pts = [(c, -r) for (r, c) in pts]
    mr = min(r for r, _ in pts)
    mc = min(c for _, c in pts)
    return [(r - mr, c - mc) for (r, c) in pts]


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); K = int(next(it))
    pieces = []   # list of (cells, cnt)
    for _ in range(K):
        A = int(next(it)); cnt = int(next(it))
        cells = [(int(next(it)), int(next(it))) for _ in range(A)]
        pieces.append((cells, cnt))
    return H, W, K, pieces


def score(instance_path, solution_path):
    H, W, K, pieces = read_instance(instance_path)

    try:
        stoks = open(solution_path).read().split()
    except Exception:
        return 0
    if not stoks:
        return 0
    it = iter(stoks)
    try:
        P = int(next(it))
    except (StopIteration, ValueError):
        return 0
    if P < 0:
        return 0

    used = [0] * K
    occupied = set()
    covered = 0
    for _ in range(P):
        try:
            k = int(next(it)); rot = int(next(it))
            ar = int(next(it)); ac = int(next(it))
        except (StopIteration, ValueError):
            return 0  # truncated or unparseable output
        if k < 0 or k >= K:
            return 0
        if rot < 0 or rot > 3:
            return 0
        used[k] += 1
        if used[k] > pieces[k][1]:
            return 0  # exceeded available count
        shape = rotate(pieces[k][0], rot)
        for (dr, dc) in shape:
            r = ar + dr
            c = ac + dc
            if r < 0 or r >= H or c < 0 or c >= W:
                return 0  # outside grid
            if (r, c) in occupied:
                return 0  # overlap
            occupied.add((r, c))
            covered += 1
    # Extra non-whitespace tokens after the declared placements are malformed.
    try:
        next(it)
        return 0
    except StopIteration:
        pass
    return covered


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION\n")
        sys.exit(2)
    print(score(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
