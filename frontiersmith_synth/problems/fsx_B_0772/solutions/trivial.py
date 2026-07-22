# TIER: trivial
"""Naive baseline: keep the input's own feature boxes as regions; for each one, use the
FIRST axis (index order 0,1,2) whose cross-section fits a stock sheet.  Never looks at
neighbours, never looks at the pin budget."""
import sys


def region_cost(dims, axis, SW, SH):
    ext = list(dims)
    count = ext[axis]
    others = [ext[i] for i in range(3) if i != axis]
    p, q = others
    orientations = []
    if p <= SW and q <= SH:
        orientations.append((p, q))
    if q <= SW and p <= SH and (q, p) != (p, q):
        orientations.append((q, p))
    if not orientations:
        return False
    for (w, h) in orientations:
        per_sheet = (SW // w) * (SH // h)
        if per_sheet >= 1:
            return True
    return False


def main():
    data = sys.stdin.read().split()
    p = 0

    def next_i():
        nonlocal p
        v = int(data[p]); p += 1
        return v

    X, Y, Z = next_i(), next_i(), next_i()
    NB = next_i()
    feats = []
    for _ in range(NB):
        b = tuple(next_i() for _ in range(6))
        feats.append(b)
    SW, SH = next_i(), next_i()
    P = next_i()

    out = [str(NB)]
    for (x0, y0, z0, x1, y1, z1) in feats:
        dims = (x1 - x0, y1 - y0, z1 - z0)
        axis = None
        for ax in range(3):
            if region_cost(dims, ax, SW, SH):
                axis = ax
                break
        if axis is None:
            axis = 0
        out.append("%d %d %d %d %d %d %d" % (x0, y0, z0, x1, y1, z1, axis))
    print("\n".join(out))


if __name__ == "__main__":
    main()
