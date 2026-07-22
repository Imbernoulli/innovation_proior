# TIER: trivial
import sys

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

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    W = int(next(it)); H = int(next(it))
    A = int(next(it))
    anchors = [int(next(it)) for _ in range(A)]
    L = int(next(it))
    loads = []
    for _ in range(L):
        r = int(next(it)); c = int(next(it)); f = int(next(it))
        loads.append((r, c, f))
    M = int(next(it)); K = int(next(it))

    # Grow one 1-wide conduit straight up from each mound's own row/column to
    # its nearest anchor, then sideways along that row to meet it -- every
    # load handled completely independently, no sharing between mounds at all.
    material = set()
    for (r_l, c_l, f_l) in loads:
        a = nearest_anchor_col(anchors, c_l)
        for cell in lpath(a, r_l, c_l):
            material.add(cell)

    cells = sorted(material)
    print(len(cells))
    out = []
    for (r, c) in cells:
        out.append(f"{r} {c}")
    if out:
        print("\n".join(out))

main()
