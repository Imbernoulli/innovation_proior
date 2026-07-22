# TIER: strong
# The insight: the workpiece IS the fixture, so plan the geometry of what
# REMAINS, not the order of what leaves. Sort waste by position along the
# corridor (column-major -- a genuine wavefront direction, not row-major
# bookkeeping), then process it in a handful of forward-looking BATCHES.
# Before each batch, look ahead across the batch's column span (plus a
# lookahead pad) and -- only if the current clamp layout would leave any of
# it unsafe -- reposition ALL FOUR clamps ONCE to a bracket that already
# covers this batch comfortably. Because the bracket is chosen with foresight
# instead of reactively per cell, a handful of repositions (not hundreds)
# keeps every removal's margin high for the whole run.
import sys, math


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


def main():
    data = sys.stdin.read().split("\n")
    head = data[0].split()
    R, C, K = int(head[0]), int(head[1]), int(head[2])
    grid = data[1:1+R]
    clamps = []
    for i in range(K):
        r, c = data[1+R+i].split()
        clamps.append([int(r), int(c)])

    waste = []
    for r in range(R):
        row = grid[r]
        for c in range(C):
            if row[c] == '#':
                waste.append((r, c))
    order = sorted(waste, key=lambda x: (x[1], x[0]))  # column-major wavefront

    N_WAVES = 4
    PAD = 3
    THRESH = 0.9
    wave_n = max(1, -(-len(order) // N_WAVES))  # ceil

    ops = []
    i = 0
    while i < len(order):
        wave = order[i:i + wave_n]
        c_lo = min(c for r, c in wave)
        c_hi = max(c for r, c in wave)
        cL = max(0, c_lo - PAD)
        cR = min(C - 1, c_hi + PAD)
        target = [[0, cL], [0, cR], [R-1, cL], [R-1, cR]]
        pts = [tuple(cl) for cl in clamps]
        need = any(hull_margin(cell, pts) < THRESH for cell in wave)
        if need:
            for k in range(4):
                if clamps[k] != target[k]:
                    ops.append(('M', k, target[k][0], target[k][1]))
                    clamps[k] = target[k]
        for cell in wave:
            ops.append(('R', cell[0], cell[1]))
        i += wave_n

    out = [str(len(ops))]
    for op in ops:
        if op[0] == 'R':
            out.append("R %d %d" % (op[1], op[2]))
        else:
            out.append("M %d %d %d" % (op[1], op[2], op[3]))
    sys.stdout.write("\n".join(out) + "\n")


main()
