#!/usr/bin/env python3
"""Trivial baseline: density greedy with NO union-overlap awareness.

Reads an instance on stdin, writes a feasible subset. The baseline ranks every
affordable antenna by raw footprint demand / cost (ignoring that overlapping
antennas double-count cells) and adds them in that fixed order while they fit.
This is the natural "toy greedy" a beginner writes; it is feasible but leaves
score on the table because it never accounts for coverage overlap. The real
solver must beat it.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    G = int(next(it)); M = int(next(it)); B = int(next(it))
    demand = [int(next(it)) for _ in range(G * G)]
    sites = []
    for _ in range(M):
        sx = int(next(it)); sy = int(next(it)); r = int(next(it)); c = int(next(it))
        sites.append((sx, sy, r, c))

    # raw footprint demand of each site (no overlap awareness)
    order = []
    for i, (sx, sy, r, c) in enumerate(sites):
        if c > B:
            continue
        x0 = max(0, sx - r); x1 = min(G - 1, sx + r)
        y0 = max(0, sy - r); y1 = min(G - 1, sy + r)
        g = 0
        for y in range(y0, y1 + 1):
            base = y * G
            for x in range(x0, x1 + 1):
                g += demand[base + x]
        order.append((g / c, i, c))
    order.sort(reverse=True)

    chosen = []
    cost = 0
    for ratio, i, c in order:
        if cost + c <= B:
            chosen.append(i)
            cost += c
    sys.stdout.write(str(len(chosen)))
    for i in chosen:
        sys.stdout.write(" " + str(i))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
