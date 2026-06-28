#!/usr/bin/env python3
"""Deterministic local scorer for Wall Painting (ALE-10).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE

The instance is an N x N target grid of colours in 0..C-1 plus an operation
budget T. The solution is a list of Q axis-aligned rectangular brush strokes.
A canvas of N x N cells starts filled with colour 0; the strokes are applied
in the given order (later strokes overwrite earlier ones); the score is the
number of canvas cells whose final colour equals the target.

SCORE  = #cells where final_canvas[r][c] == target[r][c].

FEASIBILITY -> 0 FLOOR. The score is 0 (the worst possible) if the solution is
infeasible for ANY of these reasons:
  * malformed / unparseable / truncated output;
  * Q < 0 or Q > T  (too many strokes);
  * a stroke's rectangle is degenerate or out of order (r1 > r2 or c1 > c2);
  * a stroke's rectangle leaves the N x N grid (r1<0, r2>=N, c1<0, c2>=N);
  * a stroke's colour is outside 0..C-1.
  * extra non-whitespace tokens after the declared Q strokes.
An empty stroke list (Q = 0) is feasible: the canvas stays all-0, so the score
is the number of target cells that are already colour 0 (a non-trivial floor).

The script prints a single integer (the score) to stdout.
"""
import sys


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    N = int(next(it)); C = int(next(it)); T = int(next(it))
    grid = [[int(next(it)) for _ in range(N)] for _ in range(N)]
    return N, C, T, grid


def score(instance_path, solution_path):
    N, C, T, target = read_instance(instance_path)

    try:
        stoks = open(solution_path).read().split()
    except Exception:
        return 0
    if not stoks:
        return 0
    it = iter(stoks)
    try:
        Q = int(next(it))
    except (StopIteration, ValueError):
        return 0
    if Q < 0 or Q > T:
        return 0

    # Canvas starts as colour 0 everywhere.
    canvas = [[0] * N for _ in range(N)]
    for _ in range(Q):
        try:
            r1 = int(next(it)); c1 = int(next(it))
            r2 = int(next(it)); c2 = int(next(it))
            col = int(next(it))
        except (StopIteration, ValueError):
            return 0  # truncated / malformed stroke
        if r1 > r2 or c1 > c2:
            return 0  # degenerate or mis-ordered rectangle
        if r1 < 0 or c1 < 0 or r2 >= N or c2 >= N:
            return 0  # leaves the grid
        if col < 0 or col >= C:
            return 0  # colour out of palette
        for r in range(r1, r2 + 1):
            row = canvas[r]
            for c in range(c1, c2 + 1):
                row[c] = col
    try:
        next(it)
        return 0  # trailing non-whitespace output is malformed
    except StopIteration:
        pass

    match = 0
    for r in range(N):
        tr = target[r]
        cr = canvas[r]
        for c in range(N):
            if cr[c] == tr[c]:
                match += 1
    return match


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION\n")
        sys.exit(2)
    print(score(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
