# TIER: strong
# Free the disk centres (each must still cover its platform) and run a
# deterministic push-and-refill local search:
#   * radii come from water-filling coordinate ascent given the current centres;
#   * each centre is nudged into open space (away from the nearest neighbours and
#     the nearest wall), the radii are refilled, and the move is kept ONLY if the
#     full solution stays feasible (platform cover + walls + non-overlap) and the
#     total radius does not drop. Monotone + always feasible; beats the fixed-
#     centre greedy by relocating disks toward slack.
import sys, math

TOL = 1e-6


def read_pts():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    pts = []
    idx = 1
    for _ in range(N):
        pts.append((float(toks[idx]), float(toks[idx + 1]))); idx += 2
    return N, pts


def waterfill(N, cx, cy, passes=30):
    r = [0.0] * N
    d = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d[i][j] = d[j][i] = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
    for _ in range(passes):
        for i in range(N):
            cap = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
            for j in range(N):
                if j == i:
                    continue
                room = d[i][j] - r[j]
                if room < cap:
                    cap = room
            r[i] = cap if cap > 0.0 else 0.0
    return r


def feasible(N, pts, cx, cy, r):
    for i in range(N):
        if r[i] < -TOL:
            return False
        if cx[i] - r[i] < -TOL or cx[i] + r[i] > 1.0 + TOL:
            return False
        if cy[i] - r[i] < -TOL or cy[i] + r[i] > 1.0 + TOL:
            return False
        px, py = pts[i]
        if (cx[i] - px) ** 2 + (cy[i] - py) ** 2 > r[i] * r[i] + TOL:
            return False
    for a in range(N):
        for b in range(a + 1, N):
            if math.hypot(cx[a] - cx[b], cy[a] - cy[b]) < r[a] + r[b] - TOL:
                return False
    return True


def main():
    N, pts = read_pts()
    cx = [p[0] for p in pts]
    cy = [p[1] for p in pts]
    r = waterfill(N, cx, cy)

    def total(rr):
        return sum(rr)

    cur = total(r)
    for _ in range(10):
        for i in range(N):
            # push direction: away from nearest neighbours + nearest wall
            dxsum = 0.0; dysum = 0.0
            near_d = None
            for j in range(N):
                if j == i:
                    continue
                dx = cx[i] - cx[j]; dy = cy[i] - cy[j]
                dd = math.hypot(dx, dy)
                if dd < 1e-9:
                    continue
                if near_d is None or dd < near_d:
                    near_d = dd
                w = 1.0 / (dd * dd)
                dxsum += dx / dd * w
                dysum += dy / dd * w
            # wall repulsion (push toward centre of the free interior)
            wl = cx[i]; wr = 1.0 - cx[i]; wb = cy[i]; wt = 1.0 - cy[i]
            dxsum += (1.0 / max(wl, 1e-3) - 1.0 / max(wr, 1e-3)) * 0.05
            dysum += (1.0 / max(wb, 1e-3) - 1.0 / max(wt, 1e-3)) * 0.05
            nrm = math.hypot(dxsum, dysum)
            if nrm < 1e-12:
                continue
            step = 0.5 * max(r[i], 1e-3)
            ox, oy = cx[i], cy[i]
            nx = min(1.0, max(0.0, cx[i] + dxsum / nrm * step))
            ny = min(1.0, max(0.0, cy[i] + dysum / nrm * step))
            cx[i], cy[i] = nx, ny
            r2 = waterfill(N, cx, cy)
            if feasible(N, pts, cx, cy, r2) and total(r2) >= cur - 1e-12:
                r = r2
                cur = total(r2)
            else:
                cx[i], cy[i] = ox, oy
                r = waterfill(N, cx, cy)
                cur = total(r)

    out = []
    for i in range(N):
        out.append("%.9f %.9f %.9f" % (cx[i], cy[i], r[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
