import sys, random

def nearest_anchor_col(anchors, c):
    best = None; bd = None
    for a in anchors:
        d = abs(a - c)
        if bd is None or d < bd or (d == bd and a < best):
            bd = d; best = a
    return best

def lpath(anchor_col, r_l, c_l):
    cells = []
    for r in range(1, r_l + 1):
        cells.append((r, anchor_col))
    lo, hi = min(anchor_col, c_l), max(anchor_col, c_l)
    for c in range(lo, hi + 1):
        cells.append((r_l, c))
    return cells

def baseline_cells(anchors, loads):
    s = set()
    for (r_l, c_l, f_l) in loads:
        a = nearest_anchor_col(anchors, c_l)
        for cell in lpath(a, r_l, c_l):
            s.add(cell)
    return s

# Difficulty ladder, testId 1..10. Geometry (anchor/load positions) is fixed per
# testId -- it is what plants the two required traps:
#   - cases 5,6,8,10: a later reef mound sits at a STRICTLY higher row and an
#     unaligned column relative to everything built so far, so any "connect to
#     the nearest existing point" strategy is tempted to cut a diagonal corner.
#   - all other cases keep every mound row/column-aligned with an anchor or an
#     already-built row, so a face-respecting path is free/obvious there.
# Only the force magnitudes get small testId-seeded jitter (never the geometry,
# so the planted alignment/trap structure never breaks).
CASES = [
    (12, 8,  [4],       [(6, 4, 5)]),
    (10, 12, [5],       [(10, 5, 6)]),
    (14, 9,  [3],       [(5, 3, 5), (5, 9, 5)]),
    (14, 9,  [3, 10],   [(6, 3, 5), (6, 10, 5)]),
    (16, 12, [4],       [(6, 4, 6), (10, 9, 6)]),
    (20, 14, [4],       [(6, 4, 6), (9, 10, 5), (12, 15, 4)]),
    (22, 10, [2],       [(7, 2, 5), (7, 10, 5), (7, 18, 5)]),
    (26, 16, [4],       [(7, 4, 7), (11, 12, 6), (14, 20, 5)]),
    (28, 14, [3],       [(11, 3, 7), (11, 13, 7), (11, 23, 7)]),
    (30, 18, [4],       [(8, 4, 8), (12, 14, 6), (16, 24, 5)]),
]

K_ITERS = 40
SLACK = 2.0

def main():
    i = int(sys.argv[1])
    rng = random.Random(48200 + i)
    idx = (i - 1) % len(CASES)
    W, H, anchors, loads_base = CASES[idx]

    loads = []
    for (r, c, f) in loads_base:
        jitter = rng.randint(-1, 1)
        f2 = max(1, f + jitter)
        loads.append((r, c, f2))

    bcells = baseline_cells(anchors, loads)
    M = len(bcells) + max(6, round(SLACK * len(bcells)))
    K = K_ITERS

    out = [f"{W} {H}", f"{len(anchors)}", " ".join(map(str, anchors)), f"{len(loads)}"]
    for (r, c, f) in loads:
        out.append(f"{r} {c} {f}")
    out.append(str(M))
    out.append(str(K))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
