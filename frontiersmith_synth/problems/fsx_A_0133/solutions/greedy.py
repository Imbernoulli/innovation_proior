# TIER: greedy
# Start from the equal-radius grid centres, then GROW each bubble to the largest feasible
# radius (limited by the airspace boundary, no-fly zones, and already-fixed bubbles).
# Process boundary/corner cells first so they claim the slack before interior disks.
import sys, math

TOL = 1e-6


def grid_centers(N, cx, cy, R, zones):
    gx = int(math.ceil(math.sqrt(N)))
    if gx < 1:
        gx = 1
    gy = int(math.ceil(N / float(gx)))
    cw = (2.0 * R) / gx
    ch = (2.0 * R) / gy
    r0 = min(cw, ch) * 0.5 * 0.999
    x0 = cx - R
    y0 = cy - R
    cents = []
    for j in range(gy):
        for i in range(gx):
            if len(cents) >= N:
                break
            px = x0 + (i + 0.5) * cw
            py = y0 + (j + 0.5) * ch
            if math.hypot(px - cx, py - cy) + r0 > R:
                continue
            ok = True
            for (zx, zy, zr) in zones:
                if math.hypot(px - zx, py - zy) < r0 + zr:
                    ok = False
                    break
            if ok:
                cents.append((px, py))
    return cents


def max_radius(px, py, cx, cy, R, zones, placed):
    # airspace boundary: bubble inside disk => r <= R - dist(center,C)
    r = R - math.hypot(px - cx, py - cy)
    for (zx, zy, zr) in zones:
        r = min(r, math.hypot(px - zx, py - zy) - zr)
    for (ox, oy, orr) in placed:
        r = min(r, math.hypot(px - ox, py - oy) - orr)
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

    cents = grid_centers(N, cx, cy, R, zones)
    # grow the ones nearest the airspace boundary first (they have the most slack)
    order = sorted(range(len(cents)),
                   key=lambda k: R - math.hypot(cents[k][0] - cx, cents[k][1] - cy))
    placed = []
    for k in order:
        px, py = cents[k]
        r = max_radius(px, py, cx, cy, R, zones, placed)
        if r > 0:
            placed.append((px, py, r))

    out = [str(len(placed))]
    for (x, y, r) in placed:
        out.append("%r %r %r" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
