# TIER: greedy
# Start from the equal-radius grid, then grow each diffuser to the largest feasible radius
# (limited by room walls, racks, and already-fixed disks). Beats the touching-grid baseline.
import sys, math

TOL = 1e-6


def dist_pt_rect(px, py, rect):
    x0, y0, x1, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def grid_centers(N, W, H, racks):
    gx = int(math.ceil(math.sqrt(N)))
    if gx < 1:
        gx = 1
    gy = int(math.ceil(N / float(gx)))
    cw = W / gx
    ch = H / gy
    r0 = min(cw, ch) * 0.5 * 0.999
    cents = []
    for j in range(gy):
        for i in range(gx):
            if len(cents) >= N:
                break
            cx = (i + 0.5) * cw
            cy = (j + 0.5) * ch
            if cx - r0 < 0 or cx + r0 > W or cy - r0 < 0 or cy + r0 > H:
                continue
            ok = True
            for rk in racks:
                if dist_pt_rect(cx, cy, rk) < r0:
                    ok = False
                    break
            if ok:
                cents.append((cx, cy))
    return cents


def max_radius(cx, cy, W, H, racks, placed):
    r = min(cx, W - cx, cy, H - cy)
    for rk in racks:
        r = min(r, dist_pt_rect(cx, cy, rk))
    for (ox, oy, orr) in placed:
        r = min(r, math.hypot(cx - ox, cy - oy) - orr)
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

    cents = grid_centers(N, W, H, racks)
    placed = []
    # grow interior-out isn't needed; process boundary/corner cells first for bigger gains
    order = sorted(range(len(cents)),
                   key=lambda k: min(cents[k][0], W - cents[k][0],
                                     cents[k][1], H - cents[k][1]))
    for k in order:
        cx, cy = cents[k]
        r = max_radius(cx, cy, W, H, racks, placed)
        if r > 0:
            placed.append((cx, cy, r))

    out = [str(len(placed))]
    for (x, y, r) in placed:
        out.append("%r %r %r" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
