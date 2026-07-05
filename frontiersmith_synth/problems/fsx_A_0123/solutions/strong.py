# TIER: strong
# Best-first insertion: over a fine candidate lattice, repeatedly place the single disk whose
# largest feasible radius is maximal, given the room walls, racks, and disks placed so far.
# Fills big gaps (near walls / between racks) with large disks -> large total radius.
import sys, math

TOL = 1e-6


def dist_pt_rect(px, py, rect):
    x0, y0, x1, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def max_radius(cx, cy, W, H, racks, placed):
    r = min(cx, W - cx, cy, H - cy)
    if r <= 0:
        return 0.0
    for rk in racks:
        d = dist_pt_rect(cx, cy, rk)
        if d < r:
            r = d
    for (ox, oy, orr) in placed:
        d = math.hypot(cx - ox, cy - oy) - orr
        if d < r:
            r = d
        if r <= 0:
            return 0.0
    return max(0.0, r - TOL)


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); W = float(toks[1]); H = float(toks[2]); K = int(toks[3])
    racks = []
    idx = 4
    for _ in range(K):
        racks.append((float(toks[idx]), float(toks[idx + 1]),
                      float(toks[idx + 2]), float(toks[idx + 3])))
        idx += 4

    G = 44
    cand = []
    for a in range(G):
        cx = (a + 0.5) * W / G
        for b in range(G):
            cy = (b + 0.5) * H / G
            cand.append((cx, cy))

    placed = []
    for _ in range(N):
        best_r = 0.0
        best_c = None
        for (cx, cy) in cand:
            r = max_radius(cx, cy, W, H, racks, placed)
            if r > best_r:
                best_r = r
                best_c = (cx, cy)
        if best_c is None or best_r <= TOL:
            break
        placed.append((best_c[0], best_c[1], best_r))

    out = [str(len(placed))]
    for (x, y, r) in placed:
        out.append("%r %r %r" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
