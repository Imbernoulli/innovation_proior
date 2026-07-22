# TIER: greedy
"""The obvious 'textbook' fix over trivial: pick ONE slicing axis for the WHOLE sculpture
(try all 3, keep whichever is globally feasible and cheapest) and apply it to every
region.  A single global axis never pays an alignment pin (every boundary agrees), but it
is blind to per-feature shape: a feature whose thin dimension runs a different way than
the chosen axis may not fit a stock sheet at all."""
import sys
import math


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
        return False, None, None, count, None
    best = None
    for (w, h) in orientations:
        per_sheet = (SW // w) * (SH // h)
        if per_sheet < 1:
            continue
        if best is None or per_sheet > best[2]:
            best = (w, h, per_sheet)
    if best is None:
        return False, None, None, count, None
    w, h, per_sheet = best
    return True, w, h, count, per_sheet


def pooled_sheets(feats, axis, SW, SH):
    classes = {}
    for (x0, y0, z0, x1, y1, z1) in feats:
        dims = (x1 - x0, y1 - y0, z1 - z0)
        fits, w, h, count, per_sheet = region_cost(dims, axis, SW, SH)
        if not fits:
            return False, None
        key = (w, h)
        if key not in classes:
            classes[key] = [0, per_sheet]
        classes[key][0] += count
    F = 0
    for cnt, per_sheet in classes.values():
        F += math.ceil(cnt / per_sheet)
    return True, F


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

    best_axis = None
    best_F = None
    for ax in range(3):
        feasible, F = pooled_sheets(feats, ax, SW, SH)
        if feasible and (best_F is None or F < best_F):
            best_F = F
            best_axis = ax

    if best_axis is None:
        # no single global axis works for every feature -- there is no way to save this
        # with one axis; fall back to axis 0 (the checker will correctly score it 0).
        best_axis = 0

    out = [str(NB)]
    for (x0, y0, z0, x1, y1, z1) in feats:
        out.append("%d %d %d %d %d %d %d" % (x0, y0, z0, x1, y1, z1, best_axis))
    print("\n".join(out))


if __name__ == "__main__":
    main()
