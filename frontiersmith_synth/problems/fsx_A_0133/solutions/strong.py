# TIER: strong
# Best-first insertion: over a fine candidate lattice (clipped to the airspace disk), repeatedly
# place the single bubble whose largest feasible radius is maximal given the airspace boundary,
# no-fly zones, and bubbles placed so far. Fills the biggest gaps (near the boundary, between
# no-fly zones) with large bubbles -> large total radius.
import sys, math

TOL = 1e-6


def max_radius(px, py, cx, cy, R, zones, placed):
    r = R - math.hypot(px - cx, py - cy)
    if r <= 0:
        return 0.0
    for (zx, zy, zr) in zones:
        d = math.hypot(px - zx, py - zy) - zr
        if d < r:
            r = d
        if r <= 0:
            return 0.0
    for (ox, oy, orr) in placed:
        d = math.hypot(px - ox, py - oy) - orr
        if d < r:
            r = d
        if r <= 0:
            return 0.0
    return max(0.0, r - TOL)


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); cx = float(toks[1]); cy = float(toks[2])
    R = float(toks[3]); K = int(toks[4])
    zones = []
    idx = 5
    for _ in range(K):
        zones.append((float(toks[idx]), float(toks[idx + 1]), float(toks[idx + 2])))
        idx += 3

    # fine lattice over the bounding box, keep only points inside the airspace disk
    G = 46
    x0 = cx - R
    y0 = cy - R
    cand = []
    for a in range(G):
        px = x0 + (a + 0.5) * (2.0 * R) / G
        for b in range(G):
            py = y0 + (b + 0.5) * (2.0 * R) / G
            if math.hypot(px - cx, py - cy) <= R:
                cand.append((px, py))

    placed = []
    for _ in range(N):
        best_r = 0.0
        best_c = None
        for (px, py) in cand:
            r = max_radius(px, py, cx, cy, R, zones, placed)
            if r > best_r:
                best_r = r
                best_c = (px, py)
        if best_c is None or best_r <= TOL:
            break
        placed.append((best_c[0], best_c[1], best_r))

    out = [str(len(placed))]
    for (x, y, r) in placed:
        out.append("%r %r %r" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
