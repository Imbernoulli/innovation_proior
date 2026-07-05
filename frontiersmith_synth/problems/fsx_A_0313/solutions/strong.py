# TIER: strong
# Sum-of-radii is maximized by many circles kept as large as possible, so the
# real lever is choosing the grid shape that matches the plot aspect ratio (and
# staggering rows) instead of the fixed ceil(sqrt(N)) grid the greedy uses.
# Strong sweeps every column count, builds both an aligned and a hex-staggered
# grid, grows each circle to touch the walls, keep-out obstacles, and previously
# fixed circles, and emits whichever layout gives the largest total clearance.
import sys
import math


def read_instance():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    W = float(toks[p]); p += 1
    H = float(toks[p]); p += 1
    M = int(toks[p]); p += 1
    obs = []
    for _ in range(M):
        ox = float(toks[p]); p += 1
        oy = float(toks[p]); p += 1
        oR = float(toks[p]); p += 1
        obs.append((ox, oy, oR))
    return N, W, H, obs


N, W, H, obs = read_instance()


def grow(centers):
    """Sequentially grow each center to its largest feasible radius."""
    placed = []
    for (cx, cy) in centers:
        r = min(cx, W - cx, cy, H - cy)
        if r <= 0.0:
            continue
        for (ox, oy, oR) in obs:
            r = min(r, math.hypot(cx - ox, cy - oy) - oR)
        for (px, py, pr) in placed:
            r = min(r, math.hypot(cx - px, cy - py) - pr)
        if r > 0.0:
            placed.append((cx, cy, r))
    return placed


def layout(cols, stagger):
    rows = int(math.ceil(float(N) / cols))
    cw = W / cols
    ch = H / rows
    centers = []
    count = 0
    for rr in range(rows):
        # stagger shifts alternate rows by half a cell (hex-like)
        off = (cw * 0.5) if (stagger and (rr & 1)) else 0.0
        for cc in range(cols):
            if count >= N:
                break
            cx = (cc + 0.5) * cw + off
            if cx > W:
                cx = W - (cx - W)  # reflect the overhang back inside
            cy = (rr + 0.5) * ch
            centers.append((cx, cy))
            count += 1
    return centers


def maxr(px, py, others):
    r = min(px, W - px, py, H - py)
    if r <= 0.0:
        return 0.0
    for (ox, oy, oR) in obs:
        r = min(r, math.hypot(px - ox, py - oy) - oR)
        if r <= 0.0:
            return 0.0
    for (qx, qy, qr) in others:
        r = min(r, math.hypot(px - qx, py - qy) - qr)
        if r <= 0.0:
            return 0.0
    return r


def refine(placed, rounds=6):
    P = [list(c) for c in placed]
    for _ in range(rounds):
        for i in range(len(P)):
            others = P[:i] + P[i + 1:]
            bx, by, _ = P[i]
            br = maxr(bx, by, others)
            step = min(W, H) * 0.05
            for _ in range(22):
                improved = False
                for (dx, dy) in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
                                 (step, step), (step, -step), (-step, step), (-step, -step)):
                    r = maxr(bx + dx, by + dy, others)
                    if r > br + 1e-9:
                        bx, by, br = bx + dx, by + dy, r
                        improved = True
                if not improved:
                    step *= 0.5
                if step < min(W, H) * 1e-4:
                    break
            P[i] = [bx, by, br]
    return [tuple(c) for c in P]


best = []
best_tot = -1.0
for cols in range(1, N + 1):
    for stagger in (False, True):
        placed = grow(layout(cols, stagger))
        tot = sum(c[2] for c in placed)
        if tot > best_tot + 1e-12:
            best_tot = tot
            best = placed

best = refine(best)

out = [str(len(best))]
for (x, y, r) in best:
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
