# TIER: strong
"""The insight: decouple sheet-efficiency from interface cost.  Keep the input's feature
boxes as regions (they are already the natural per-part decomposition) but let EACH
region pick its own feasible axis independently, and spend the alignment-pin budget where
it actually buys something.  This is exactly the isoperimetric trade the family is built
around: paying a boundary cost (pins, at the few region-region interfaces) to unlock an
interior win (fewer, better-packed sheets) that a single global axis can never reach and
that a per-feature-only optimum (ignoring neighbours) does not need -- until the sheet
math makes matching a neighbour's axis the cheaper way to stay within budget.

We brute-force over the (small) product of each region's own feasible axes -- exact, not
a heuristic -- because the region count here is always tiny; the "insight" is the
reformulation (decouple pins from packing, search jointly) not the search procedure."""
import sys
import math
import itertools


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


def feasible_axes(dims, SW, SH):
    out = []
    for ax in range(3):
        fits, *_ = region_cost(dims, ax, SW, SH)
        if fits:
            out.append(ax)
    return out


def pooled_sheets(dims_axis_list, SW, SH):
    classes = {}
    for dims, axis in dims_axis_list:
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


def shared_face_area(b1, b2):
    x0, y0, z0, x1, y1, z1 = b1
    X0, Y0, Z0, X1, Y1, Z1 = b2
    total = 0
    if x1 == X0 or X1 == x0:
        oy = max(0, min(y1, Y1) - max(y0, Y0))
        oz = max(0, min(z1, Z1) - max(z0, Z0))
        if oy > 0 and oz > 0:
            total += oy * oz
    if y1 == Y0 or Y1 == y0:
        ox = max(0, min(x1, X1) - max(x0, X0))
        oz = max(0, min(z1, Z1) - max(z0, Z0))
        if ox > 0 and oz > 0:
            total += ox * oz
    if z1 == Z0 or Z1 == z0:
        ox = max(0, min(x1, X1) - max(x0, X0))
        oy = max(0, min(y1, Y1) - max(y0, Y0))
        if ox > 0 and oy > 0:
            total += ox * oy
    return total


def pins_for(feats, combo):
    total = 0
    n = len(feats)
    for i in range(n):
        for j in range(i + 1, n):
            if combo[i] == combo[j]:
                continue
            total += shared_face_area(feats[i], feats[j])
    return total


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
    dims_list = []
    for _ in range(NB):
        b = tuple(next_i() for _ in range(6))
        feats.append(b)
        dims_list.append((b[3] - b[0], b[4] - b[1], b[5] - b[2]))
    SW, SH = next_i(), next_i()
    P = next_i()

    per_feat_axes = [feasible_axes(d, SW, SH) for d in dims_list]

    best = None  # (F, pins, combo)
    for combo in itertools.product(*per_feat_axes):
        feasible, F = pooled_sheets(list(zip(dims_list, combo)), SW, SH)
        if not feasible:
            continue
        pins_used = pins_for(feats, combo)
        if pins_used > P:
            continue
        key = (F, pins_used)
        if best is None or key < (best[0], best[1]):
            best = (F, pins_used, combo)

    if best is None:
        # should not happen if the instance is well-formed; fall back to axis 0 everywhere
        combo = tuple(0 for _ in feats)
    else:
        combo = best[2]

    out = [str(NB)]
    for (x0, y0, z0, x1, y1, z1), ax in zip(feats, combo):
        out.append("%d %d %d %d %d %d %d" % (x0, y0, z0, x1, y1, z1, ax))
    print("\n".join(out))


if __name__ == "__main__":
    main()
