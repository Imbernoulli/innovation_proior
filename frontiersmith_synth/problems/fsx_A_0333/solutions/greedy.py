# TIER: greedy
# Keep every centre on its platform (feasibility trivially guaranteed by radii)
# but replace the symmetric half-split with WATER-FILLING coordinate ascent:
# repeatedly regrow each radius to the largest value allowed by the walls and by
# the already-set neighbour radii. This unequal allocation beats the even split.
import sys, math


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    pts = []
    idx = 1
    for _ in range(N):
        pts.append((float(toks[idx]), float(toks[idx + 1]))); idx += 2

    bd = [min(x, 1.0 - x, y, 1.0 - y) for (x, y) in pts]
    d = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            dij = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            d[i][j] = d[j][i] = dij

    # symmetric baseline allocation (feasible)
    base = []
    for i in range(N):
        best = bd[i]
        for j in range(N):
            if j == i:
                continue
            if 0.5 * d[i][j] < best:
                best = 0.5 * d[i][j]
        base.append(best if best > 0.0 else 0.0)

    # water-filling coordinate ascent (feasible, unequal split)
    r = [0.0] * N
    for _ in range(50):
        for i in range(N):
            cap = bd[i]
            for j in range(N):
                if j == i:
                    continue
                room = d[i][j] - r[j]
                if room < cap:
                    cap = room
            if cap < 0.0:
                cap = 0.0
            r[i] = cap

    # keep whichever feasible fixed-centre allocation totals more
    r = r if sum(r) >= sum(base) else base

    out = []
    for i in range(N):
        out.append("%.9f %.9f %.9f" % (pts[i][0], pts[i][1], r[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
