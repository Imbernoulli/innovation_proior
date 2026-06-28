#!/usr/bin/env python3
"""Independent brute-force oracle for the domino-tiling-count problem.

Reads:  h w p
        h lines, each a string of '.'/'#' of length w
Writes: number of ways to tile ALL '.' cells exactly with 1x2 dominoes
        (each domino covers two orthogonally adjacent '.' cells; '#' never covered),
        taken modulo p.

Method: plain recursive backtracking. Find the first uncovered free cell in
row-major order; try to extend it right or down with a domino; recurse.
Obviously correct, exponential -- only used on tiny grids.
"""
import sys


def solve(h, w, p, grid):
    free = [[grid[r][c] == '.' for c in range(w)] for r in range(h)]
    covered = [[False] * w for _ in range(h)]
    count = 0

    def find_next():
        for r in range(h):
            for c in range(w):
                if free[r][c] and not covered[r][c]:
                    return (r, c)
        return None

    def rec():
        nonlocal count
        cell = find_next()
        if cell is None:
            count += 1
            return
        r, c = cell
        # Option A: horizontal domino (r,c)-(r,c+1)
        if c + 1 < w and free[r][c + 1] and not covered[r][c + 1]:
            covered[r][c] = covered[r][c + 1] = True
            rec()
            covered[r][c] = covered[r][c + 1] = False
        # Option B: vertical domino (r,c)-(r+1,c)
        if r + 1 < h and free[r + 1][c] and not covered[r + 1][c]:
            covered[r][c] = covered[r + 1][c] = True
            rec()
            covered[r][c] = covered[r + 1][c] = False
        # else: cell (r,c) cannot be covered -> dead end, contributes nothing

    rec()
    return count % p


def main():
    data = sys.stdin.read().split()
    idx = 0
    h = int(data[idx]); idx += 1
    w = int(data[idx]); idx += 1
    p = int(data[idx]); idx += 1
    grid = []
    for _ in range(h):
        grid.append(data[idx]); idx += 1
    print(solve(h, w, p, grid))


if __name__ == "__main__":
    main()
