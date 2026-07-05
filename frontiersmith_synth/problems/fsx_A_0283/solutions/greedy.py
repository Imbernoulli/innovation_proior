# TIER: greedy
# Coarse largest-empty-circle greedy. Over a coarse candidate grid, repeatedly place the
# pad whose centre admits the largest feasible radius (limited by the four walls, every
# forbidden zone, and all previously placed pads). Produces boundary-hugging unequal pads
# that beat the equal bottom row, but the coarse grid leaves quality on the table.
import sys, math


def solve():
    t = sys.stdin.read().split()
    N = int(t[0]); S = float(t[1]); K = int(t[2])
    zones = []
    p = 3
    for _ in range(K):
        zones.append((float(t[p]), float(t[p + 1]), float(t[p + 2]))); p += 3

    G = 22
    cands = []
    for i in range(G):
        for j in range(G):
            cx = (i + 0.5) * S / G
            cy = (j + 0.5) * S / G
            cands.append((cx, cy))

    placed = []  # (x, y, r)

    def feasible_radius(cx, cy):
        r = min(cx, S - cx, cy, S - cy)
        for (ox, oy, g) in zones:
            d = math.hypot(cx - ox, cy - oy) - g
            if d < r:
                r = d
        for (px, py, pr) in placed:
            d = math.hypot(cx - px, cy - py) - pr
            if d < r:
                r = d
        return r

    for _ in range(N):
        best = 0.0; bx = by = None
        for (cx, cy) in cands:
            rr = feasible_radius(cx, cy)
            if rr > best:
                best = rr; bx, by = cx, cy
        if bx is None or best <= 1e-9:
            break
        placed.append((bx, by, best))

    out = [str(len(placed))]
    for (x, y, r) in placed:
        out.append("%.10f %.10f %.10f" % (x, y, r * (1.0 - 1e-6)))
    sys.stdout.write("\n".join(out) + "\n")


solve()
