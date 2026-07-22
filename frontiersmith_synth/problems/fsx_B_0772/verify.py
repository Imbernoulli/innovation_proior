#!/usr/bin/env python3
"""Checker for the crossed-plywood slab decomposition problem (format C).

CLI: python3 verify.py <in> <out> <ans>   (ans is an unused placeholder)
Prints a line ending in `Ratio: <float in [0,1]>` and exits 0.
"""
import sys
import math

MAX_REGIONS = 30
COORD_CAP = 2000  # defensive upper bound on any single coordinate token


def fail(reason):
    print("INFEASIBLE: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    with open(path, "r") as f:
        return f.read().split()


def parse_int_strict(tok):
    """Reject anything that isn't a clean base-10 integer (nan/inf/floats/junk)."""
    try:
        if not isinstance(tok, str) or len(tok) == 0:
            return None
        s = tok
        if s[0] in "+-":
            s2 = s[1:]
        else:
            s2 = s
        if len(s2) == 0 or not s2.isdigit():
            return None
        return int(tok)
    except Exception:
        return None


def region_cost(dims, axis, SW, SH):
    """dims=(dx,dy,dz). Returns (fits, w, h, count, per_sheet) with the SAME deterministic
    orientation/tie-break rule everywhere (used for both the submission and the baseline)."""
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


def pooled_sheets(region_axis_dims):
    """region_axis_dims: list of (dims, axis). Returns (feasible, F) pooling identical-size
    slab classes across regions before applying the ceiling."""
    classes = {}
    for dims, axis, SW, SH in region_axis_dims:
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


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---------- parse input ----------
    itoks = read_tokens(in_path)
    ip = 0

    def next_i():
        nonlocal ip
        v = int(itoks[ip])
        ip += 1
        return v

    X, Y, Z = next_i(), next_i(), next_i()
    NB = next_i()
    feats = []
    for _ in range(NB):
        b = tuple(next_i() for _ in range(6))
        feats.append(b)
    SW, SH = next_i(), next_i()
    P = next_i()

    # ---------- build occupied grid ----------
    if X <= 0 or Y <= 0 or Z <= 0 or X * Y * Z > 4_000_000:
        fail("bad grid")
    occ = bytearray(X * Y * Z)

    def idx(x, y, z):
        return (x * Y + y) * Z + z

    total_solid = 0
    for (x0, y0, z0, x1, y1, z1) in feats:
        for x in range(x0, x1):
            base_x = x * Y
            for y in range(y0, y1):
                base = (base_x + y) * Z
                for z in range(z0, z1):
                    ii = base + z
                    if occ[ii] == 0:
                        occ[ii] = 1
                        total_solid += 1

    # ---------- parse output (defensively) ----------
    otoks = read_tokens(out_path)
    if len(otoks) == 0:
        fail("empty output")
    op = 0
    R_raw = parse_int_strict(otoks[op]) if op < len(otoks) else None
    if R_raw is None:
        fail("bad region count")
    op += 1
    if R_raw < 1 or R_raw > MAX_REGIONS:
        fail("region count out of range")
    R = R_raw

    expected_tokens = 1 + 7 * R
    if len(otoks) != expected_tokens:
        fail("wrong token count")

    regions = []
    for _ in range(R):
        vals = []
        for _ in range(7):
            v = parse_int_strict(otoks[op])
            op += 1
            if v is None:
                fail("non-integer / nan / inf token")
            if abs(v) > COORD_CAP:
                fail("coordinate out of range")
            vals.append(v)
        x0, y0, z0, x1, y1, z1, axis = vals
        if axis not in (0, 1, 2):
            fail("axis must be 0, 1 or 2")
        if not (0 <= x0 < x1 <= X and 0 <= y0 < y1 <= Y and 0 <= z0 < z1 <= Z):
            fail("region box out of grid bounds or degenerate")
        regions.append((x0, y0, z0, x1, y1, z1, axis))

    # ---------- coverage: regions must exactly partition the solid ----------
    cover = bytearray(X * Y * Z)
    covered_cells = 0
    for (x0, y0, z0, x1, y1, z1, axis) in regions:
        for x in range(x0, x1):
            base_x = x * Y
            for y in range(y0, y1):
                base = (base_x + y) * Z
                for z in range(z0, z1):
                    ii = base + z
                    if occ[ii] == 0:
                        fail("region covers a non-solid cell")
                    if cover[ii] != 0:
                        fail("regions overlap")
                    cover[ii] = 1
                    covered_cells += 1
    if covered_cells != total_solid:
        fail("regions do not cover the whole solid")

    # ---------- per-region fit check ----------
    region_axis_dims = []
    for (x0, y0, z0, x1, y1, z1, axis) in regions:
        dims = (x1 - x0, y1 - y0, z1 - z0)
        fits, w, h, count, per_sheet = region_cost(dims, axis, SW, SH)
        if not fits:
            fail("a slab does not fit on any stock sheet under its assigned axis")
        region_axis_dims.append((dims, axis, SW, SH))

    # ---------- pin budget ----------
    pins_used = 0
    for i in range(R):
        bi = regions[i][:6]
        ai = regions[i][6]
        for j in range(i + 1, R):
            bj = regions[j][:6]
            aj = regions[j][6]
            if ai == aj:
                continue
            pins_used += shared_face_area(bi, bj)
    if pins_used > P:
        fail("alignment-pin budget exceeded (%d > %d)" % (pins_used, P))

    # ---------- objective F (submission) ----------
    feasible, F = pooled_sheets(region_axis_dims)
    if not feasible:
        fail("internal packing failure")

    # ---------- internal baseline B: per-feature axis = smallest index that fits ----------
    b_rad = []
    for (x0, y0, z0, x1, y1, z1) in feats:
        dims = (x1 - x0, y1 - y0, z1 - z0)
        chosen = None
        for ax in range(3):
            fits, w, h, count, per_sheet = region_cost(dims, ax, SW, SH)
            if fits:
                chosen = ax
                break
        if chosen is None:
            fail("instance error: no baseline axis fits (generator bug)")
        b_rad.append((dims, chosen, SW, SH))
    _, B = pooled_sheets(b_rad)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("pins_used=%d P=%d F=%d B=%d" % (pins_used, P, F, B))
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
