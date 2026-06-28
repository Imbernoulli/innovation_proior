#!/usr/bin/env python3
"""Instance generator for Grid Polyomino Packing (ALE-02).

Usage: python3 gen.py SEED  > instance.txt

Instance format (stdin of the solver):
    H W K
    then K piece-type blocks; block k is:
        A_k cnt_k
        A_k lines, each "r c" : the cells of the base orientation of piece k,
                                given as non-negative offsets (min row = min col = 0).

A "piece type" is a fixed polyomino. The solver may place up to cnt_k copies of
type k, each independently rotated by 0/90/180/270 degrees, anywhere on the grid,
as long as all placed cells are inside the HxW grid and no two placed cells of any
two placements coincide (placements may not overlap).

The generator draws a board size and a small library of random connected
polyominoes (sizes 1..6) plus generous per-type counts, so that the board cannot
be fully covered in general -> a genuine optimization gap exists.
"""
import sys
import random


def random_polyomino(rng, target):
    """Grow a connected polyomino of `target` cells by random accretion."""
    cells = {(0, 0)}
    frontier = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while len(cells) < target and frontier:
        idx = rng.randrange(len(frontier))
        cell = frontier.pop(idx)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cell[0] + dr, cell[1] + dc)
            if nb not in cells:
                frontier.append(nb)
    # normalise so min row = min col = 0
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return sorted(((r - mr, c - mc) for r, c in cells))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed * 1_000_003 + 12345)

    # Board: moderate sizes so a strong solver and the baseline differ clearly.
    H = rng.randint(12, 30)
    W = rng.randint(12, 30)

    # Piece library: 4..8 distinct types, areas 1..6.
    K = rng.randint(4, 8)
    pieces = []
    seen = set()
    attempts = 0
    while len(pieces) < K and attempts < 200:
        attempts += 1
        target = rng.randint(2, 6)
        poly = tuple(random_polyomino(rng, target))
        if poly in seen:
            continue
        seen.add(poly)
        pieces.append(list(poly))
    K = len(pieces)

    # Per-type counts: generous, so supply usually exceeds what fits ->
    # the board is the binding constraint, not the inventory.
    board_area = H * W
    out = [f"{H} {W} {K}"]
    for poly in pieces:
        area = len(poly)
        # enough copies to (over)fill the board on their own
        cnt = max(1, board_area // area // 2 + rng.randint(0, 3))
        out.append(f"{area} {cnt}")
        for r, c in poly:
            out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
