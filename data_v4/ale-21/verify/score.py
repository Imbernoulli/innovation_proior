#!/usr/bin/env python3
"""Deterministic local scorer for "Guillotine Cutting Stock".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). HIGHER is better.

Scoring rule (see context.md "Evaluation settings"):
  * Instance: n requested rectangles and an unlimited supply of W x H sheets.
  * SOLUTION format:
        m
        idx sheet x y rot      (m lines)
    Each line places requested rectangle `idx` (0-based, into the input list) on
    sheet number `sheet` (>= 0) with its bottom-left corner at integer (x, y).
    rot is 0 (use w x h) or 1 (rotated, use h x w). A requested rectangle may be
    omitted (left unplaced); it may NOT appear twice.

  * FEASIBILITY (any violation -> score 0):
      - the file parses as the integer m followed by exactly m well-formed lines;
      - every idx is in [0, n) and appears at most once;
      - every rot in {0, 1};
      - each placed rectangle lies fully inside its sheet:
            0 <= x, x + pw <= W  and  0 <= y, y + ph <= H
        where (pw, ph) is (w, h) or (h, w) per rot;
      - within every sheet the placed rectangles are pairwise non-overlapping AND
        the placement is GUILLOTINE-LEGAL: the occupied set on the sheet can be
        separated by a sequence of edge-to-edge (full-width or full-height) cuts,
        each cut not crossing the interior of any placed rectangle, recursively,
        until every part holds at most one rectangle.
    If any of these fail, the whole solution is INFEASIBLE and scores 0.

  * COST (lower is better) of a feasible solution:
        S       = number of distinct sheets that contain >= 1 placed rectangle
        placed  = total area of placed rectangles
        unplaced= total area of rectangles left out
        P       = UNPLACED_PENALTY (a fixed constant, see below)
        cost    = S * W * H - placed + P * unplaced
    i.e. every opened sheet costs its full area W*H, recovered by the area we
    actually placed, plus a penalty P per unit of unplaced requested area.

  * SCORE (higher better), normalized against the deterministic shelf next-fit
    baseline the scorer recomputes itself:
        score = round(1_000_000 * baseline_cost / max(1, solver_cost))
    The baseline scores ~1_000_000; a lower-cost (better) packing scores more.
    INFEASIBLE -> 0.
"""
import sys

UNPLACED_PENALTY = 3  # P: cost per unit area of an unplaced rectangle


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    W = int(next(it))
    H = int(next(it))
    rects = []
    for _ in range(n):
        w = int(next(it))
        h = int(next(it))
        rects.append((w, h))
    return n, W, H, rects


def read_solution(path, n, W, H, rects):
    """Parse + fully validate. Return list of placements or None if infeasible.

    A placement is a tuple (idx, sheet, x, y, pw, ph).
    """
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if not toks:
        return None
    it = iter(toks)
    try:
        m = int(next(it))
    except (StopIteration, ValueError):
        return None
    if m < 0 or m > n:
        return None
    placements = []
    used_idx = set()
    for _ in range(m):
        try:
            idx = int(next(it))
            sheet = int(next(it))
            x = int(next(it))
            y = int(next(it))
            rot = int(next(it))
        except (StopIteration, ValueError):
            return None
        if idx < 0 or idx >= n or idx in used_idx:
            return None
        used_idx.add(idx)
        if sheet < 0:
            return None
        if rot not in (0, 1):
            return None
        w, h = rects[idx]
        pw, ph = (w, h) if rot == 0 else (h, w)
        if x < 0 or y < 0 or x + pw > W or y + ph > H:
            return None
        placements.append((idx, sheet, x, y, pw, ph))
    # Reject trailing garbage tokens.
    if next(it, None) is not None:
        return None
    return placements


# ----------------------------------------------------------------- geometry checks
def overlap(a, b):
    """Do two axis-aligned rects (x, y, pw, ph) overlap with positive area?"""
    _, _, ax, ay, aw, ah = a
    _, _, bx, by, bw, bh = b
    if ax + aw <= bx or bx + bw <= ax:
        return False
    if ay + ah <= by or by + bh <= ay:
        return False
    return True


def guillotine_ok(items):
    """items: list of (x, y, pw, ph) relative to the whole sheet.

    Return True iff the set is separable by recursive edge-to-edge cuts. Each
    recursion is restricted to the bounding box of `items`; we look for a full
    cut (vertical or horizontal) spanning that box that does not pass through the
    interior of any rectangle and that has rectangles strictly on both sides.
    """
    if len(items) <= 1:
        return True
    xs = [it[0] for it in items]
    ys = [it[1] for it in items]
    xe = [it[0] + it[2] for it in items]
    ye = [it[1] + it[3] for it in items]
    x0, x1 = min(xs), max(xe)
    y0, y1 = min(ys), max(ye)

    # Candidate vertical cut lines: rectangle right/left edges strictly inside.
    cand_v = sorted({it[0] for it in items} | {it[0] + it[2] for it in items})
    for cx in cand_v:
        if cx <= x0 or cx >= x1:
            continue
        # A rectangle crosses the line if its interior straddles cx.
        crosses = False
        left = []
        right = []
        for it in items:
            ix0, ix1 = it[0], it[0] + it[2]
            if ix0 < cx < ix1:
                crosses = True
                break
            if ix1 <= cx:
                left.append(it)
            else:
                right.append(it)
        if crosses or not left or not right:
            continue
        return guillotine_ok(left) and guillotine_ok(right)

    cand_h = sorted({it[1] for it in items} | {it[1] + it[3] for it in items})
    for cy in cand_h:
        if cy <= y0 or cy >= y1:
            continue
        crosses = False
        down = []
        up = []
        for it in items:
            iy0, iy1 = it[1], it[1] + it[3]
            if iy0 < cy < iy1:
                crosses = True
                break
            if iy1 <= cy:
                down.append(it)
            else:
                up.append(it)
        if crosses or not down or not up:
            continue
        return guillotine_ok(down) and guillotine_ok(up)

    return False


def feasible_and_cost(placements, n, W, H, rects):
    """Return (True, cost) if feasible, else (False, None)."""
    by_sheet = {}
    placed_area = 0
    placed_idx = set()
    for p in placements:
        idx, sheet, x, y, pw, ph = p
        by_sheet.setdefault(sheet, []).append(p)
        placed_area += pw * ph
        placed_idx.add(idx)

    for sheet, items in by_sheet.items():
        # pairwise non-overlap
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if overlap(items[i], items[j]):
                    return False, None
        # guillotine separability
        rel = [(x, y, pw, ph) for (_, _, x, y, pw, ph) in items]
        if not guillotine_ok(rel):
            return False, None

    S = len(by_sheet)
    total_area = sum(w * h for (w, h) in rects)
    unplaced_area = total_area - placed_area
    cost = S * W * H - placed_area + UNPLACED_PENALTY * unplaced_area
    return True, cost


# -------------------------------------------------------------- baseline: next-fit
def baseline_cost(n, W, H, rects):
    """Deterministic shelf-based next-fit packing.

    Rectangles in input order; orient so the longer side is the height (without
    exceeding bounds, preferring the orientation that fits). Pack left-to-right
    into the current shelf (a horizontal band of the current sheet). When a rect
    does not fit the shelf width, open a new shelf above; when it does not fit the
    remaining sheet height, open a new sheet. Shelf packing is always guillotine
    legal (cut between shelves, then between rects in a shelf). Rectangles that do
    not fit even on a fresh sheet are left unplaced.
    """
    sheets = 0
    placed_area = 0
    cur_x = 0
    shelf_h = 0
    used_h = 0
    have_sheet = False

    def orient(w, h):
        # choose an orientation that fits the sheet; prefer the one given.
        cands = []
        if w <= W and h <= H:
            cands.append((w, h))
        if h <= W and w <= H:
            cands.append((h, w))
        if not cands:
            return None
        # prefer smaller height to pack shelves tightly, tie-break smaller width
        cands.sort(key=lambda c: (c[1], c[0]))
        return cands[0]

    for (w, h) in rects:
        o = orient(w, h)
        if o is None:
            continue  # cannot ever be placed
        pw, ph = o
        if not have_sheet:
            sheets = 1
            have_sheet = True
            cur_x = 0
            shelf_h = 0
            used_h = 0
        # try current shelf
        if cur_x + pw <= W and used_h + max(shelf_h, ph) <= H:
            # fits if it does not push the shelf past the sheet top
            if cur_x + pw <= W and used_h + ph <= H:
                placed_area += pw * ph
                cur_x += pw
                shelf_h = max(shelf_h, ph)
                continue
        # try a new shelf on the same sheet
        if used_h + shelf_h + ph <= H:
            used_h += shelf_h
            cur_x = 0
            shelf_h = ph
            if pw <= W:
                placed_area += pw * ph
                cur_x = pw
                continue
        # open a new sheet
        sheets += 1
        cur_x = 0
        shelf_h = ph
        used_h = 0
        if pw <= W and ph <= H:
            placed_area += pw * ph
            cur_x = pw

    total_area = sum(w * h for (w, h) in rects)
    unplaced_area = total_area - placed_area
    cost = sheets * W * H - placed_area + UNPLACED_PENALTY * unplaced_area
    return cost


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, W, H, rects = read_instance(sys.argv[1])

    placements = read_solution(sys.argv[2], n, W, H, rects)
    if placements is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    ok, cost = feasible_and_cost(placements, n, W, H, rects)
    if not ok:
        print(0)
        return

    base = baseline_cost(n, W, H, rects)
    score = int(round(1_000_000.0 * base / max(1, cost)))
    print(score)


if __name__ == "__main__":
    main()
