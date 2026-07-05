import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

# ---------- shared polyomino helpers ----------
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

def greedy_pack(W, H, blocked, cands):
    # cands: list of (tid, rels, stock) in priority order
    occ = set(blocked)
    used = {c[0]: 0 for c in cands}
    placements = []
    for y in range(H):
        for x in range(W):
            if (x, y) in occ:
                continue
            for (tid, rels, stock) in cands:
                if used[tid] >= stock:
                    continue
                done = False
                for rel in rels:
                    cells = [(x + rx, y + ry) for (rx, ry) in rel]
                    ok = True
                    for (cx, cy) in cells:
                        if cx < 0 or cx >= W or cy < 0 or cy >= H or (cx, cy) in occ:
                            ok = False; break
                    if ok:
                        for cc in cells:
                            occ.add(cc)
                        placements.append((tid, cells))
                        used[tid] += 1
                        done = True
                        break
                if done:
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
        bedrock = set()
        for _ in range(R):
            ox = int(next(it)); oy = int(next(it))
            bedrock.add((ox, oy))
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

    # canonical orientation sets (bbox-normalized) for matching
    orient_sets = [set(transforms(sh)) for sh in shapes]

    # ---- internal baseline B: anchored greedy with type 0 only ----
    rels0 = orient_rels(shapes[0])
    base_pl = greedy_pack(W, H, bedrock, [(0, rels0, stocks[0])])
    B = sum(len(cells) for (_, cells) in base_pl)
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
        # in-grid, not bedrock, distinct within panel
        seen_local = set()
        for (cx, cy) in cells:
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                fail("cell out of grid (%d,%d)" % (cx, cy))
            if (cx, cy) in bedrock:
                fail("cell on bedrock (%d,%d)" % (cx, cy))
            if (cx, cy) in seen_local:
                fail("duplicate cell within panel")
            seen_local.add((cx, cy))
        if len(seen_local) != s:
            fail("panel cell count mismatch")
        # legal copy of type t?
        if normalize(cells) not in orient_sets[t]:
            fail("cells are not a rotation/reflection of type %d" % t)
        # no overlap with previously placed
        for c in cells:
            if c in occ:
                fail("overlap at (%d,%d)" % c)
        for c in cells:
            occ.add(c)
        used[t] += 1
        if used[t] > stocks[t]:
            fail("stock exceeded for type %d" % t)
        F += s

    sc = min(1000.0, 100.0 * F / max(1, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
