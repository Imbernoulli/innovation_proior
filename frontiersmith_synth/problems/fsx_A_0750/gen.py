#!/usr/bin/env python3
"""
gen.py <testId> -- one pocket-clearing instance per testId (1..10), deterministic.

Layout (all seeded ONLY from testId):
  - A row of vertical "chambers", each a solid block of a DISTINCT width equal to one of
    the catalog's tool sizes (excluding 1) -- exercising multi-radius-coverage: the right
    tool for each chamber is dictated by its width, not by taste.
  - Chambers are separated by single-column vertical checkerboard "barriers": a barrier
    column alternates pocket/non-pocket every row, so ANY square of size >= 2 that would
    overlap it always straddles one non-pocket cell in that column -- barriers are
    PROVABLY size-1-only, modelling the forced small-tool finishing pass near a tight
    (concave) feature ("corner-rest" cells) that no amount of cleverness can avoid.
  - A few 1-wide comb "teeth" hang off the bottom edge: more unavoidable size-1 corridor,
    and more chances for a naive scan to interleave tool sizes.
"""
import sys, random


def build(test_id: int):
    Lmax = [3, 3, 5, 5, 7, 7, 9, 9, 9, 9][test_id - 1]
    catalog = list(range(1, Lmax + 1, 2))          # e.g. [1,3,5,7,9]
    sizes_no1 = catalog[1:]                         # chamber widths (multi-radius menu)
    n_chambers = [2, 3, 3, 4, 4, 5, 5, 6, 7, 8][test_id - 1]
    H = [12, 14, 20, 24, 28, 30, 35, 40, 44, 50][test_id - 1]
    rng = random.Random(5000 + test_id)

    seq = [sizes_no1[i % len(sizes_no1)] for i in range(n_chambers)]
    rng.shuffle(seq)

    cols = []   # ('chamber', width) or ('barrier',)
    for i, w in enumerate(seq):
        cols.append(('chamber', w))
        if i != len(seq) - 1:
            cols.append(('barrier', 0))

    W = 0
    for kind, w in cols:
        W += w if kind == 'chamber' else 1

    grid = [['#'] * W for _ in range(H)]
    c = 0
    for kind, w in cols:
        if kind == 'chamber':
            c += w
        else:
            for r in range(H):
                if r % 2 == 1:
                    grid[r][c] = '.'
            c += 1

    n_teeth = max(1, n_chambers // 2)
    tooth_len = max(2, max(sizes_no1) - 1)
    grid = grid + [['.'] * W for _ in range(tooth_len)]
    for i in range(n_teeth):
        tc = (i * max(1, W // n_teeth) + 1) % W
        for k in range(tooth_len):
            grid[H + k][tc] = '#'
    Htot = H + tooth_len

    C = 2 + (test_id % 4)
    return Htot, W, catalog, C, grid


def main():
    test_id = int(sys.argv[1])
    H, W, catalog, C, grid = build(test_id)
    out = []
    out.append(f"{H} {W} {len(catalog)} {C}")
    out.append(" ".join(str(s) for s in catalog))
    for r in range(H):
        out.append("".join(grid[r]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
