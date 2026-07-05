# TIER: strong
# Two constructions, take the better total:
#   (a) fine largest-empty-circle greedy on a dense candidate grid -- finds bigger empty
#       disks than the coarse greedy, especially in the clear outer margin;
#   (b) a hexagonal equal-radius lattice, dropping sites that collide with a forbidden zone,
#       binary-searched for the largest radius that still seats N pads.
# The two behave differently per instance, and the max-of-two dominates the coarse greedy.
import sys, math


def read_instance():
    t = sys.stdin.read().split()
    N = int(t[0]); S = float(t[1]); K = int(t[2])
    zones = []
    p = 3
    for _ in range(K):
        zones.append((float(t[p]), float(t[p + 1]), float(t[p + 2]))); p += 3
    return N, S, K, zones


def fine_greedy(N, S, zones):
    G = 48
    cands = [((i + 0.5) * S / G, (j + 0.5) * S / G) for i in range(G) for j in range(G)]
    placed = []

    def feas(cx, cy):
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
            rr = feas(cx, cy)
            if rr > best:
                best = rr; bx, by = cx, cy
        if bx is None or best <= 1e-9:
            break
        placed.append((bx, by, best))
    return placed


def hex_sites(r, S, zones):
    if r <= 0:
        return []
    dy = math.sqrt(3.0) * r
    pts = []
    row = 0
    y = r
    while y <= S - r + 1e-12:
        offset = r if (row % 2 == 1) else 0.0
        x = r + offset
        while x <= S - r + 1e-12:
            ok = True
            for (ox, oy, g) in zones:
                if math.hypot(x - ox, y - oy) < r + g - 1e-12:
                    ok = False; break
            if ok:
                pts.append((x, y))
            x += 2.0 * r
        y += dy
        row += 1
    return pts


def hex_pack(N, S, zones):
    lo, hi = 1e-9, S / 2.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if len(hex_sites(mid, S, zones)) >= N:
            lo = mid
        else:
            hi = mid
    r = lo
    pts = hex_sites(r, S, zones)
    return [(x, y, r) for (x, y) in pts[:N]]


def total(pl):
    return sum(p[2] for p in pl)


def main():
    N, S, K, zones = read_instance()
    a = fine_greedy(N, S, zones)
    b = hex_pack(N, S, zones)
    placed = a if total(a) >= total(b) else b

    out = [str(len(placed))]
    for (x, y, r) in placed:
        out.append("%.10f %.10f %.10f" % (x, y, r * (1.0 - 1e-6)))
    sys.stdout.write("\n".join(out) + "\n")


main()
