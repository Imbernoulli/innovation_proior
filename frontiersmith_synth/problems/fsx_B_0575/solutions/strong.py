# TIER: strong

import sys

# ---- orientation: choose which of the 6 axis directions points DOWN (build/gravity). ----
# oriented frame (u,v,w) with w = build height (w=0 == build plate).
def oriented_dims(X, Y, Z, o):
    if o in (0, 1): return X, Y, Z
    if o in (2, 3): return X, Z, Y
    return Y, Z, X

def to_oriented(x, y, z, X, Y, Z, o):
    if o == 0: return (x, y, z)
    if o == 1: return (x, y, Z - 1 - z)
    if o == 2: return (x, z, y)
    if o == 3: return (x, z, Y - 1 - y)
    if o == 4: return (y, z, x)
    return (y, z, X - 1 - x)

def from_oriented(u, v, w, X, Y, Z, o):
    if o == 0: return (u, v, w)
    if o == 1: return (u, v, Z - 1 - w)
    if o == 2: return (u, w, v)
    if o == 3: return (u, Y - 1 - w, v)
    if o == 4: return (w, u, v)
    return (X - 1 - w, u, v)

def build_solid_grid(solids, X, Y, Z, o):
    U, V, W = oriented_dims(X, Y, Z, o)
    grid = [[[0] * U for _ in range(V)] for _ in range(W)]
    for (x, y, z) in solids:
        u, v, w = to_oriented(x, y, z, X, Y, Z, o)
        grid[w][v][u] = 1
    return grid, U, V, W

# ---- tool-access line of sight: a cell can be reached/extracted iff a straight axis ray
# reaches the bounding-box exterior through only non-solid cells.  Allowed ray directions in
# the oriented frame: +-u, +-v (lateral) and +w (up).  -w (down) is blocked by the plate. ----
def escapable_grid(grid, U, V, W):
    esc = [[[False] * U for _ in range(V)] for _ in range(W)]
    for w in range(W):
        gw = grid[w]; ew = esc[w]
        for v in range(V):
            row = gw[v]; er = ew[v]
            clear = True                      # +u  (toward u=U boundary)
            for u in range(U - 1, -1, -1):
                if row[u] == 0:
                    if clear: er[u] = True
                else:
                    clear = False
            clear = True                      # -u  (toward u=0 boundary)
            for u in range(U):
                if row[u] == 0:
                    if clear: er[u] = True
                else:
                    clear = False
    for w in range(W):
        gw = grid[w]; ew = esc[w]
        for u in range(U):
            clear = True                      # +v
            for v in range(V - 1, -1, -1):
                if gw[v][u] == 0:
                    if clear: ew[v][u] = True
                else:
                    clear = False
            clear = True                      # -v
            for v in range(V):
                if gw[v][u] == 0:
                    if clear: ew[v][u] = True
                else:
                    clear = False
    for v in range(V):                        # +w  (up)
        for u in range(U):
            clear = True
            for w in range(W - 1, -1, -1):
                if grid[w][v][u] == 0:
                    if clear: esc[w][v][u] = True
                else:
                    clear = False
    return esc

def strict_support(grid, U, V, W):
    """Straight vertical columns: fill every empty cell beneath an overhang down to the
    first solid or the plate (the obvious minimal-material support for one orientation)."""
    sup = set()
    for u in range(U):
        for v in range(V):
            for w in range(1, W):
                if grid[w][v][u] == 1 and grid[w - 1][v][u] == 0:
                    ww = w - 1
                    while ww >= 0 and grid[ww][v][u] == 0:
                        sup.add((u, v, ww)); ww -= 1
    return sup

def score_support(grid, esc, U, V, W, osup, LAM):
    vol = len(osup)
    uc = 0
    for (u, v, w) in osup:
        if not esc[w][v][u]:
            uc += 1
    return vol + LAM * uc

def evaluate(X, Y, Z, LAM, solids, o, support):
    """Return (feasible, F). F = support volume + LAM * (# unremovable support voxels)."""
    if o is None or o < 0 or o > 5:
        return (False, 0)
    supset = set()
    for (x, y, z) in support:
        if not (0 <= x < X and 0 <= y < Y and 0 <= z < Z): return (False, 0)
        if (x, y, z) in solids: return (False, 0)
        if (x, y, z) in supset: return (False, 0)
        supset.add((x, y, z))
    grid, U, V, W = build_solid_grid(solids, X, Y, Z, o)
    osup = set(to_oriented(x, y, z, X, Y, Z, o) for (x, y, z) in supset)
    def occ(u, v, w):
        return grid[w][v][u] == 1 or (u, v, w) in osup
    # facet rule for solids: strict cell directly below must be filled
    for w in range(1, W):
        gw = grid[w]
        for v in range(V):
            row = gw[v]
            for u in range(U):
                if row[u] == 1 and not occ(u, v, w - 1):
                    return (False, 0)
    # supports may slope (<=45 deg): a supporter within the 3x3 below
    for (u, v, w) in osup:
        if w == 0: continue
        good = False
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                nu = u + du; nv = v + dv
                if 0 <= nu < U and 0 <= nv < V and occ(nu, nv, w - 1):
                    good = True; break
            if good: break
        if not good:
            return (False, 0)
    esc = escapable_grid(grid, U, V, W)
    return (True, score_support(grid, esc, U, V, W, osup, LAM))

def baseline_F(solids, X, Y, Z, LAM):
    grid, U, V, W = build_solid_grid(solids, X, Y, Z, 0)
    sup = strict_support(grid, U, V, W)
    esc = escapable_grid(grid, U, V, W)
    return max(1, score_support(grid, esc, U, V, W, sup, LAM))

def routed_support(grid, U, V, W, LAM):
    """Insight solver's core: route each mandatory support (directly under an overhang)
    DOWN via <=45deg slopes, minimising per-voxel cost 1 + LAM*[not escapable].
    Escapability depends on SOLIDS only (support is transparent to the extraction ray),
    so it is a fixed field -> a clean per-seed min-cost descent, shared by union."""
    esc = escapable_grid(grid, U, V, W)
    INF = float('inf')
    best = [[[INF] * U for _ in range(V)] for _ in range(W)]
    par = [[[None] * U for _ in range(V)] for _ in range(W)]
    for w in range(W):
        for v in range(V):
            for u in range(U):
                if grid[w][v][u] == 1:
                    continue
                cc = 1 + (0 if esc[w][v][u] else LAM)
                if w == 0:
                    best[w][v][u] = cc; par[w][v][u] = 'P'; continue
                options = []
                for du in (-1, 0, 1):
                    for dv in (-1, 0, 1):
                        nu = u + du; nv = v + dv
                        if 0 <= nu < U and 0 <= nv < V:
                            if grid[w - 1][nv][nu] == 1:
                                options.append((0, 'S'))
                            else:
                                c = best[w - 1][nv][nu]
                                if c < INF:
                                    options.append((c, (nu, nv, w - 1)))
                if options:
                    childcost, pp = min(options, key=lambda t: t[0])
                    best[w][v][u] = cc + childcost; par[w][v][u] = pp
    support = set()
    for w in range(1, W):
        for v in range(V):
            for u in range(U):
                if grid[w][v][u] == 1 and grid[w - 1][v][u] == 0:
                    cu, cv, cw = u, v, w - 1
                    while True:
                        if (cu, cv, cw) in support:
                            break
                        support.add((cu, cv, cw))
                        p = par[cw][cv][cu]
                        if p in ('S', 'P', None):
                            break
                        cu, cv, cw = p
    return support, esc

def parse_instance(text):
    lines = text.split('\n')
    idx = 0
    while idx < len(lines) and lines[idx].strip() == '':
        idx += 1
    p = lines[idx].split()
    X, Y, Z, LAM = int(p[0]), int(p[1]), int(p[2]), int(p[3])
    idx += 1
    rows = []
    for ln in lines[idx:]:
        s = ln.strip()
        if s == '':
            continue
        rows.append(s)
        if len(rows) >= Y * Z:
            break
    solids = set()
    r = 0
    for s in rows[:Y * Z]:
        z = r // Y; y = r % Y
        for x, ch in enumerate(s):
            if ch == '1':
                solids.add((x, y, z))
        r += 1
    return X, Y, Z, LAM, solids

def emit(o, osup, X, Y, Z):
    coords = [from_oriented(u, v, w, X, Y, Z, o) for (u, v, w) in osup]
    out = [str(o), str(len(coords))]
    for (x, y, z) in coords:
        out.append("%d %d %d" % (x, y, z))
    sys.stdout.write("\n".join(out) + "\n")

def main():
    # insight: score orientations by ESCAPABLE shadow volume (support volume + extraction
    # penalty), and route supports along <=45deg slopes toward line-of-sight corridors.
    X, Y, Z, LAM, solids = parse_instance(sys.stdin.read())
    best = None
    for o in range(6):
        grid, U, V, W = build_solid_grid(solids, X, Y, Z, o)
        sup, esc = routed_support(grid, U, V, W, LAM)
        F = score_support(grid, esc, U, V, W, sup, LAM)
        if best is None or F < best[0]:
            best = (F, o, sup)
    _, o, sup = best
    emit(o, sup, X, Y, Z)

if __name__ == "__main__":
    main()
