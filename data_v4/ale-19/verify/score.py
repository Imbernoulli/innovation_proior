#!/usr/bin/env python3
"""Deterministic local scorer for "2D Rectangle Strip Packing".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. HIGHER is better. INFEASIBLE -> 0.

Scoring rule (see context.md "Evaluation settings"):

  * Instance:
        n W R
        w_i h_i        (n lines)
    A vertical strip of integer width W and unbounded height. R is the rotation
    flag (1 = a rectangle may be rotated 90 degrees, 0 = fixed orientation).

  * SOLUTION format (read from <solution_file>): exactly n lines, the i-th line
        x_i y_i r_i
    placing rectangle i (input order) with its bottom-left corner at integer
    (x_i, y_i). r_i in {0,1} is the rotation bit: r_i == 0 uses (w_i, h_i),
    r_i == 1 uses (h_i, w_i). Every rectangle must be placed (no omissions).

  * FEASIBILITY (any violation -> score 0):
      - the file parses as exactly n well-formed lines of three integers;
      - if R == 0 then every r_i == 0;
      - r_i in {0, 1};
      - each rectangle lies fully inside the strip:
            0 <= x_i  and  x_i + pw_i <= W  and  0 <= y_i
        where (pw_i, ph_i) = (w_i, h_i) if r_i == 0 else (h_i, w_i);
      - the n placed rectangles are pairwise non-overlapping (touching edges are
        allowed; only positive-area overlap is forbidden).
    If any of these fail, the whole solution is INFEASIBLE and scores 0.

  * HEIGHT (lower is better) of a feasible solution:
        height = max over i of (y_i + ph_i)          (0 if n == 0)

  * SCORE (higher better), normalized against the deterministic first-fit
    decreasing-height (FFDH) shelf baseline the scorer recomputes itself:
        score = round(1_000_000 * baseline_height / max(1, solver_height))
    The FFDH baseline scores ~1_000_000; a shorter (better) packing scores more.
    INFEASIBLE -> 0.
"""
import sys


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    W = int(next(it))
    R = int(next(it))
    rects = []
    for _ in range(n):
        w = int(next(it))
        h = int(next(it))
        rects.append((w, h))
    return n, W, R, rects


def read_solution(path, n, W, R, rects):
    """Parse + fully validate. Return list of (x, y, pw, ph) or None if infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    # exactly 3*n integer tokens
    if len(toks) != 3 * n:
        return None
    placed = []
    for i in range(n):
        try:
            x = int(toks[3 * i + 0])
            y = int(toks[3 * i + 1])
            r = int(toks[3 * i + 2])
        except ValueError:
            return None
        if r not in (0, 1):
            return None
        if R == 0 and r != 0:
            return None
        w, h = rects[i]
        pw, ph = (w, h) if r == 0 else (h, w)
        if x < 0 or y < 0 or x + pw > W:
            return None
        placed.append((x, y, pw, ph))
    return placed


# ----------------------------------------------------------------- geometry checks
def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if ax + aw <= bx or bx + bw <= ax:
        return False
    if ay + ah <= by or by + bh <= ay:
        return False
    return True


def feasible_height(placed):
    """Return (True, height) if pairwise non-overlapping, else (False, None).

    A sweep over y-intervals would be faster, but n is small; we do an honest
    O(n^2) pairwise check so the feasibility test is unambiguous.
    """
    n = len(placed)
    for i in range(n):
        for j in range(i + 1, n):
            if overlap(placed[i], placed[j]):
                return False, None
    height = 0
    for (x, y, pw, ph) in placed:
        height = max(height, y + ph)
    return True, height


# --------------------------------------------------------- baseline: FFDH shelves
def baseline_height(n, W, R, rects):
    """First-Fit Decreasing-Height shelf packing.

    Orient every rectangle so its (placed) height is the smaller side when that
    still fits the width W (only relevant if R == 1; otherwise keep orientation),
    sort by decreasing height, then place left-to-right into the first shelf
    whose remaining width admits the rectangle, opening a new shelf on top when
    none fits. Shelf packing never overlaps, so this is always feasible, and its
    top is a legitimate height to normalize against.
    """
    items = []
    for (w, h) in rects:
        pw, ph = w, h
        if R == 1:
            # prefer the orientation with the SMALLER height (flatter shelves),
            # provided the width still fits the strip.
            cands = []
            if w <= W:
                cands.append((w, h))
            if h <= W:
                cands.append((h, w))
            if cands:
                cands.sort(key=lambda c: (c[1], c[0]))
                pw, ph = cands[0]
        items.append((pw, ph))
    # decreasing height
    order = sorted(range(n), key=lambda i: (-items[i][1], -items[i][0]))

    shelves = []  # each shelf: [y_bottom, height, used_width]
    total_height = 0
    for i in order:
        pw, ph = items[i]
        placed = False
        for sh in shelves:
            if sh[2] + pw <= W and ph <= sh[1]:
                sh[2] += pw
                placed = True
                break
        if not placed:
            shelves.append([total_height, ph, pw])
            total_height += ph
    return total_height


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, W, R, rects = read_instance(sys.argv[1])

    placed = read_solution(sys.argv[2], n, W, R, rects)
    if placed is None:
        print(0)
        return

    ok, height = feasible_height(placed)
    if not ok:
        print(0)
        return

    if n == 0 or height == 0:
        # degenerate; nothing to pack -> perfect
        print(1_000_000)
        return

    base = baseline_height(n, W, R, rects)
    score = int(round(1_000_000.0 * base / max(1, height)))
    print(score)


if __name__ == "__main__":
    main()
