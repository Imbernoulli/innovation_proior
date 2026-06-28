#!/usr/bin/env python3
"""Trivial baseline: 'honor every target' coloring.

Each cell outputs its pinned value if pinned, else its target t[i]. This is always
FEASIBLE (pins honored, all bits) but pays a large interface penalty along every
target blob boundary -- the coloring the heuristic must beat.

Usage: python3 baseline.py <instance_file>   (writes solution grid to stdout)
"""
import sys


def main():
    with open(sys.argv[1]) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it)); _W = int(next(it))
    rows = []
    for r in range(N):
        row = []
        for c in range(N):
            _h = int(next(it)); t = int(next(it)); p = int(next(it))
            row.append(str(p if p != -1 else t))
        rows.append(" ".join(row))
    sys.stdout.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
