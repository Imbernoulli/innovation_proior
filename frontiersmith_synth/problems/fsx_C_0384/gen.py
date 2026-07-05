#!/usr/bin/env python3
"""gen.py <testId> -- print ONE freight-yard block-assignment instance to stdout.

An n x n board of (track, shift) cells. A cut for exactly one destination block
(0..n-1) is built per cell. Two rules:
  * Latin: a block appears at most once per row and once per column.
  * Forbidden list: each cell (i,j) has a set Fb[i][j] of blocks that may NOT be
    built there (loading-gauge / weight incompatibilities).
Some cells are pre-built givens. The instance is planted from a random Latin
square L (so a full restricted completion exists), but finding a large completion
is a list-restricted Latin completion problem (NP-hard). Everything is seeded by
testId only -> bit-for-bit reproducible.
"""
import sys, random


def random_latin(n, rng):
    # cyclic base, then row/col/symbol permutations preserve the Latin property.
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    rp = list(range(n)); rng.shuffle(rp)
    cp = list(range(n)); rng.shuffle(cp)
    sp = list(range(n)); rng.shuffle(sp)
    return [[sp[base[rp[i]][cp[j]]] for j in range(n)] for i in range(n)]


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260702 + 100003 * tid)

    # difficulty ladder: (n, given_density, max_forbidden_per_cell). small: n in [4..9].
    specs = [
        (4, 0.45, 1), (5, 0.44, 2), (6, 0.42, 2), (6, 0.48, 3),
        (7, 0.44, 3), (7, 0.50, 3), (8, 0.44, 3), (8, 0.50, 4),
        (9, 0.44, 4), (9, 0.50, 4),
    ]
    n, dens, fmax = specs[(tid - 1) % len(specs)]

    L = random_latin(n, rng)

    # forbidden lists: forbid up to fmax blocks per cell, never L[i][j] (keeps L a
    # valid completion, so a full board is achievable in principle but hard to find).
    forb = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            k = rng.randint(0, fmax)
            choices = [v for v in range(n) if v != L[i][j]]
            rng.shuffle(choices)
            forb[i][j] = set(choices[:k])

    # reveal givens
    target = max(1, int(round(dens * n * n)))
    cells = [(i, j) for i in range(n) for j in range(n)]
    rng.shuffle(cells)
    given = [[False] * n for _ in range(n)]
    for idx in range(min(target, len(cells))):
        i, j = cells[idx]
        given[i][j] = True

    out = [str(n)]
    for i in range(n):
        out.append(" ".join(str(L[i][j]) if given[i][j] else "." for j in range(n)))
    for i in range(n):
        row = []
        for j in range(n):
            if forb[i][j]:
                row.append(",".join(str(v) for v in sorted(forb[i][j])))
            else:
                row.append("-")
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
