import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

# ---------- polyomino helpers ----------
def normalize(cells):
    mnx = min(x for x, y in cells); mny = min(y for x, y in cells)
    return frozenset((x - mnx, y - mny) for x, y in cells)

def transforms(cells):
    res, seen = [], set()
    cur = list(cells)
    for ref in range(2):
        c = [(-x, y) for x, y in cur] if ref else list(cur)
        for _ in range(4):
            s = normalize(c)
            if s not in seen:
                seen.add(s); res.append(s)
            c = [(y, -x) for x, y in c]
    return res

def anchored(orient):
    ax, ay = min(orient, key=lambda p: (p[1], p[0]))
    return [(x - ax, y - ay) for (x, y) in orient]

def orient_rels(cells):
    return [anchored(o) for o in transforms(cells)]

def greedy_type0(W, H, blocked, rels0, stock):
    occ = set(blocked)
    used = 0
    placements = []
    for y in range(H):
        for x in range(W):
            if used >= stock:
                return placements
            if (x, y) in occ:
                continue
            for rel in rels0:
                cells = [(x + rx, y + ry) for (rx, ry) in rel]
                ok = True
                for (cx, cy) in cells:
                    if cx < 0 or cx >= W or cy < 0 or cy >= H or (cx, cy) in occ:
                        ok = False; break
                if ok:
                    for cc in cells:
                        occ.add(cc)
                    placements.append(cells)
                    used += 1
                    break
    return placements


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse input ----
    try:
        it = iter(inp)
        W = int(next(it)); H = int(next(it))
        R = int(next(it))
        blocked = set()
        for _ in range(R):
            ox = int(next(it)); oy = int(next(it))
            blocked.add((ox, oy))
        w = {}
        for y in range(H):
            for x in range(W):
                w[(x, y)] = int(next(it))
        P = int(next(it))
        sizes = []; shapes = []; stocks = []
        for _ in range(P):
            c = int(next(it)); s = int(next(it))
            cells = []
            for _ in range(s):
                dx = int(next(it)); dy = int(next(it))
                cells.append((dx, dy))
            stocks.append(c); sizes.append(s); shapes.append(cells)
    except Exception:
        fail("bad input")

    orient_sets = [set(transforms(sh)) for sh in shapes]

    # ---- internal baseline B: anchored greedy with type 0 only, weighted ----
    rels0 = orient_rels(shapes[0])
    base_pl = greedy_type0(W, H, blocked, rels0, stocks[0])
    B = 0
    for cells in base_pl:
        for (cx, cy) in cells:
            B += w[(cx, cy)]
    B = max(1, B)

    # ---- parse & validate participant output ----
    try:
        oit = iter(out)
        M = int(next(oit))
    except Exception:
        fail("parse M")
    if M < 0:
        fail("M<0")

    occ = set()
    used = [0] * P
    F = 0
    for _ in range(M):
        try:
            t = int(next(oit))
        except Exception:
            fail("parse type")
        if t < 0 or t >= P:
            fail("type out of range %d" % t)
        s = sizes[t]
        cells = []
        try:
            for _ in range(s):
                cx = int(next(oit)); cy = int(next(oit))
                cells.append((cx, cy))
        except Exception:
            fail("parse cells")
        seen_local = set()
        for (cx, cy) in cells:
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                fail("cell out of grid (%d,%d)" % (cx, cy))
            if (cx, cy) in blocked:
                fail("cell on locked intersection (%d,%d)" % (cx, cy))
            if (cx, cy) in seen_local:
                fail("duplicate cell within cabinet")
            seen_local.add((cx, cy))
        if len(seen_local) != s:
            fail("cabinet cell count mismatch")
        if normalize(cells) not in orient_sets[t]:
            fail("cells are not a rotation/reflection of type %d" % t)
        for c in cells:
            if c in occ:
                fail("overlap at (%d,%d)" % c)
        for c in cells:
            occ.add(c)
        used[t] += 1
        if used[t] > stocks[t]:
            fail("stock exceeded for type %d" % t)
        for (cx, cy) in cells:
            F += w[(cx, cy)]

    sc = min(1000.0, 100.0 * F / max(1, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
