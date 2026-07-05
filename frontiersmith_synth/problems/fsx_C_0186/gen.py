#!/usr/bin/env python3
"""gen.py <testId> -- print ONE wind-farm turbine-layout instance to stdout.

An n x n array of turbine slots. To suppress wake interference, no turbine
MODEL (0..n-1) may repeat within any row or any column -> a (partial) Latin
square. Some slots are already pre-installed (givens). Difficulty grows with
testId (grid size + given density). Everything is seeded by testId only, so
instances are bit-for-bit reproducible.
"""
import sys, random


def main():
    tid = int(sys.argv[1])
    random.seed(20260701 + 7919 * tid)

    # difficulty ladder: (n, given_density). "small" scale => n in [4..9].
    specs = [
        (4, 0.45), (5, 0.42), (6, 0.42), (6, 0.50), (7, 0.45),
        (7, 0.50), (8, 0.45), (8, 0.50), (9, 0.45), (9, 0.50),
    ]
    n, dens = specs[(tid - 1) % len(specs)]
    target = max(1, int(round(dens * n * n)))

    grid = [[-1] * n for _ in range(n)]
    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]

    cells = [(i, j) for i in range(n) for j in range(n)]
    random.shuffle(cells)

    placed = 0
    for (i, j) in cells:
        if placed >= target:
            break
        avail = [v for v in range(n) if v not in rows[i] and v not in cols[j]]
        if not avail:
            continue
        v = random.choice(avail)
        grid[i][j] = v
        rows[i].add(v)
        cols[j].add(v)
        placed += 1

    out = [str(n)]
    for i in range(n):
        out.append(" ".join("." if grid[i][j] == -1 else str(grid[i][j])
                            for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
