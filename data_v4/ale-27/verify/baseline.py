#!/usr/bin/env python3
"""Trivial feasible baseline for "Grid Light Placement": place one light at the
first floor cell of every maximal HORIZONTAL corridor. This lights every floor
cell (each cell's H-corridor has a light) and uses exactly B lights, so it scores
exactly 1_000_000 under score.py. It is the floor the SA solver must beat.

Usage: python3 baseline.py <instance_file>  -> writes a solution to stdout.
"""
import sys


def main():
    with open(sys.argv[1]) as f:
        lines = f.read().split("\n")
    idx = 0
    while lines[idx].strip() == "":
        idx += 1
    H, W = map(int, lines[idx].split())
    idx += 1
    grid = []
    for r in range(H):
        row = lines[idx + r]
        if len(row) < W:
            row = row + "." * (W - len(row))
        grid.append(row[:W])

    lights = []
    for r in range(H):
        c = 0
        while c < W:
            if grid[r][c] == '.':
                lights.append((r, c))            # one light per H-corridor
                while c < W and grid[r][c] == '.':
                    c += 1
            else:
                c += 1

    out = [str(len(lights))]
    out.extend(f"{r} {c}" for (r, c) in lights)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
