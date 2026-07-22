import sys, math

OFFSET = 6.5      # fixed additive constant folded into the objective (see statement)
MAX_OPS = 200000  # hard sanity cap on the number of operations accepted


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---------------- geometry: signed distance to the convex hull of the clamps ----------------

def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_seg_dist(p, a, b):
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx-ax, by-ay
    if dx == 0 and dy == 0:
        return math.hypot(px-ax, py-ay)
    t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx+dy*dy)
    t = max(0.0, min(1.0, t))
    qx, qy = ax+t*dx, ay+t*dy
    return math.hypot(px-qx, py-qy)


def hull_margin(p, points):
    """Signed distance from point p to the convex hull of `points`.
    Positive if p is strictly inside the hull, negative if outside,
    0 on the boundary. Degenerates gracefully for collinear/short point sets."""
    hull = convex_hull(points)
    if len(hull) == 0:
        return 0.0
    if len(hull) == 1:
        return -math.hypot(p[0]-hull[0][0], p[1]-hull[0][1])
    if len(hull) == 2:
        return -point_seg_dist(p, hull[0], hull[1])
    n = len(hull)
    best = None
    for i in range(n):
        u = hull[i]; w = hull[(i+1) % n]
        ex, ey = w[0]-u[0], w[1]-u[1]
        cross = ex*(p[1]-u[1]) - ey*(p[0]-u[0])
        length = math.hypot(ex, ey)
        d = cross/length
        if best is None or d < best:
            best = d
    return best


# ---------------- instance parsing ----------------

def parse_instance(path):
    text = open(path).read().split("\n")
    head = text[0].split()
    R, C, K = int(head[0]), int(head[1]), int(head[2])
    PEN = float(head[3])
    grid = text[1:1+R]
    clamps0 = []
    for i in range(K):
        r, c = text[1+R+i].split()
        clamps0.append((int(r), int(c)))
    waste = set()
    keep = set()
    for r in range(R):
        row = grid[r]
        for c in range(C):
            if row[c] == '#':
                waste.add((r, c))
            else:
                keep.add((r, c))
    return R, C, K, PEN, waste, keep, clamps0


# ---------------- replay engine: shared by the participant's output AND the
# internal baseline (trivial) construction ----------------

def replay(R, C, K, PEN, waste, keep, clamps0, ops):
    """ops: list of ('R',r,c) or ('M',k,r,c). Returns (F, ok, reason)."""
    clamps = [list(p) for p in clamps0]
    removed = set()
    moves = 0
    margins = []
    for op in ops:
        if op[0] == 'R':
            _, r, c = op
            if (r, c) not in waste:
                return None, False, "removed a non-waste cell"
            if (r, c) in removed:
                return None, False, "cell already removed"
            if any(tuple(cl) == (r, c) for cl in clamps):
                return None, False, "cell occupied by a clamp"
            pts = [tuple(cl) for cl in clamps]
            margins.append(hull_margin((r, c), pts))
            removed.add((r, c))
        else:
            _, k, r, c = op
            if k < 0 or k >= K:
                return None, False, "bad clamp id"
            if not (0 <= r < R and 0 <= c < C):
                return None, False, "target out of bounds"
            is_keep = (r, c) in keep
            is_waste_alive = (r, c) in waste and (r, c) not in removed
            if not (is_keep or is_waste_alive):
                return None, False, "target cell does not currently exist"
            if any(j != k and tuple(clamps[j]) == (r, c) for j in range(K)):
                return None, False, "target occupied by another clamp"
            clamps[k] = [r, c]
            moves += 1
    if removed != waste:
        return None, False, "not all waste cells were removed"
    if not margins:
        return None, False, "no removals performed"
    F = (min(margins) + OFFSET) - PEN * moves
    return F, True, ""


# ---------------- internal baseline: single-clamp reactive snap ----------------
# Whenever the next cell (processed in row-major order) has margin < 0 under the
# CURRENT clamp layout, snap the near-side pair of clamps (0,1 if row 0 is
# nearer, else 2,3) to distinct columns around it. Cheap, always-feasible,
# geometrically-blind -- exactly what solutions/trivial.py reproduces.

def _place(clamps, idx, target, K, ops):
    """Move clamp idx to target=[row,col], bumping any clamp currently
    sitting there to a free column on the same row first (never collides)."""
    if clamps[idx] == target:
        return
    trow, tc = target
    for j in range(K):
        if j != idx and clamps[j] == target:
            used = {clamps[x][1] for x in range(K)}
            tmp = 0
            while tmp in used or tmp == tc:
                tmp += 1
            ops.append(('M', j, trow, tmp))
            clamps[j] = [trow, tmp]
    ops.append(('M', idx, trow, tc))
    clamps[idx] = [trow, tc]


def compute_baseline_F(R, C, K, PEN, waste, keep, clamps0):
    clamps = [list(p) for p in clamps0]
    order = sorted(waste, key=lambda x: (x[0], x[1]))
    ops = []
    for (r, c) in order:
        pts = [tuple(cl) for cl in clamps]
        m = hull_margin((r, c), pts)
        if m < 0.0:
            near_top = r <= (R - 1) - r
            colA = max(0, c - 1)
            colB = min(C - 1, c + 1)
            trow = 0 if near_top else R - 1
            idx0, idx1 = (0, 1) if near_top else (2, 3)
            _place(clamps, idx0, [trow, colA], K, ops)
            _place(clamps, idx1, [trow, colB], K, ops)
        ops.append(('R', r, c))
    F, ok, reason = replay(R, C, K, PEN, waste, keep, clamps0, ops)
    return F if ok else None


# ---------------- main ----------------

def main():
    R, C, K, PEN, waste, keep, clamps0 = parse_instance(sys.argv[1])

    out_text = open(sys.argv[2]).read()
    tokens = out_text.split()
    idx = 0
    try:
        T = int(tokens[idx]); idx += 1
    except Exception:
        fail("bad or missing op count")
    if T < 0 or T > MAX_OPS:
        fail("op count out of range")

    ops = []
    try:
        for _ in range(T):
            tag = tokens[idx]; idx += 1
            if tag == 'R':
                r = int(tokens[idx]); idx += 1
                c = int(tokens[idx]); idx += 1
                ops.append(('R', r, c))
            elif tag == 'M':
                k = int(tokens[idx]); idx += 1
                r = int(tokens[idx]); idx += 1
                c = int(tokens[idx]); idx += 1
                ops.append(('M', k, r, c))
            else:
                fail("bad op tag")
    except IndexError:
        fail("truncated output")
    except ValueError:
        fail("non-integer token")

    F, ok, reason = replay(R, C, K, PEN, waste, keep, clamps0, ops)
    if not ok:
        fail(reason)
    if not math.isfinite(F):
        fail("non-finite objective")

    B = compute_baseline_F(R, C, K, PEN, waste, keep, clamps0)
    if B is None or not math.isfinite(B) or B <= 1e-9:
        fail("internal baseline degenerate")

    sc = min(1000.0, 100.0 * F / B)
    sc = max(0.0, sc)
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
