# TIER: greedy
# The obvious "keep it safe right now" recipe: process waste cells in
# row-major order (row by row, left to right). Whenever the upcoming cell's
# margin drops below a comfortable threshold, immediately snap ALL FOUR
# clamps into a tight bracket around just that one cell (no lookahead, no
# batching). This is locally safe, but because row-major order re-approaches
# the corridor from the near edge on every new row, and the bracket only ever
# covers a couple of columns, it pays for a full four-clamp reposition over
# and over -- the refixture bill explodes on long/deep corridors even though
# every individual removal looked "safe" at the time.
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
    order = sorted(waste, key=lambda x: (x[0], x[1]))

    PAD = 1
    THRESH = 0.9
    ops = []
    for (r, c) in order:
        pts = [tuple(cl) for cl in clamps]
        m = hull_margin((r, c), pts)
        if m < THRESH:
            cl_ = max(0, c - PAD)
            cr_ = min(C - 1, c + PAD)
            target = [[0, cl_], [0, cr_], [R-1, cl_], [R-1, cr_]]
            for k in range(4):
                if clamps[k] != target[k]:
                    ops.append(('M', k, target[k][0], target[k][1]))
                    clamps[k] = target[k]
        ops.append(('R', r, c))

    out = [str(len(ops))]
    for op in ops:
        if op[0] == 'R':
            out.append("R %d %d" % (op[1], op[2]))
        else:
            out.append("M %d %d %d" % (op[1], op[2], op[3]))
    sys.stdout.write("\n".join(out) + "\n")


main()
