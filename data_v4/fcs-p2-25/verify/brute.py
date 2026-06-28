#!/usr/bin/env python3
"""Independent brute-force oracle for 'minimum falling path sum'.

A falling path starts at any cell in row 0, and from cell (i, j) moves to one of
(i+1, j-1), (i+1, j), (i+1, j+1) staying inside the grid, until it reaches the
last row. We want the minimum total of cell values along such a path.

This oracle enumerates EVERY falling path explicitly (depth-first over all start
columns and all 3-way choices), with no dynamic programming, so it is an
independent check on the O(n^2) DP in sol.cpp.
"""
import sys


def solve(n, grid):
    if n == 0:
        return 0

    best = [None]

    def dfs(row, col, acc):
        acc += grid[row][col]
        if row == n - 1:
            if best[0] is None or acc < best[0]:
                best[0] = acc
            return
        for dc in (-1, 0, 1):
            nc = col + dc
            if 0 <= nc < n:
                dfs(row + 1, nc, acc)

    for start in range(n):
        dfs(0, start, 0)

    return best[0]


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    grid = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(data[idx])); idx += 1
        grid.append(row)
    print(solve(n, grid))


if __name__ == "__main__":
    main()
