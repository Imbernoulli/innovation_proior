#!/usr/bin/env python3
"""Deterministic local scorer for "Heat-Diffusion Tile Coloring" (ALE-Bench).

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. Higher is better.

Scoring rule (must match context.md exactly):

  A solution is an N x N grid of binary coatings x[r][c] in {0,1}. It is FEASIBLE iff:
    (1) it parses as exactly N rows of N tokens, each token 0 or 1;
    (2) every PINNED cell (p != -1 in the instance) has x[r][c] == p.

  For a feasible coloring the energy functional is
        E = W * (# of 4-adjacent pairs whose coatings differ)
          + sum_cells  h[r][c] * [ x[r][c] != t[r][c] ]
  (both terms are non-negative integers; E >= 0).

  The score is
        score = round( 1e9 / (1 + E) )        if the solution is FEASIBLE,
        score = 0                              otherwise (the feasibility -> 0 floor).

  Lower energy => higher score; a perfect E = 0 coloring scores 1e9. Any infeasible
  output (wrong shape, a non-binary token, or a violated pin) floors the score to 0.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it))
    W = int(next(it))
    field = [[0] * N for _ in range(N)]
    target = [[0] * N for _ in range(N)]
    pin = [[-1] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            field[r][c] = int(next(it))
            target[r][c] = int(next(it))
            pin[r][c] = int(next(it))
    return N, W, field, target, pin


def read_solution(path, N):
    """Parse the solution grid. Returns (grid) or None if it does not parse as
    exactly N*N binary tokens."""
    with open(path) as f:
        toks = f.read().split()
    if len(toks) != N * N:
        return None
    grid = [[0] * N for _ in range(N)]
    k = 0
    for r in range(N):
        for c in range(N):
            t = toks[k]
            k += 1
            if t not in ("0", "1"):
                return None
            grid[r][c] = int(t)
    return grid


def energy(N, W, field, target, grid):
    E = 0
    # interface term: horizontal and vertical adjacent differing pairs
    for r in range(N):
        for c in range(N):
            v = grid[r][c]
            if c + 1 < N and grid[r][c + 1] != v:
                E += W
            if r + 1 < N and grid[r + 1][c] != v:
                E += W
    # field term
    for r in range(N):
        for c in range(N):
            if grid[r][c] != target[r][c]:
                E += field[r][c]
    return E


def score(instance_path, solution_path):
    N, W, field, target, pin = read_instance(instance_path)
    grid = read_solution(solution_path, N)
    if grid is None:
        return 0
    # feasibility: pins honored
    for r in range(N):
        for c in range(N):
            if pin[r][c] != -1 and grid[r][c] != pin[r][c]:
                return 0
    E = energy(N, W, field, target, grid)
    return round(1e9 / (1 + E))


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance_file> <solution_file>\n")
        sys.exit(1)
    print(score(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
