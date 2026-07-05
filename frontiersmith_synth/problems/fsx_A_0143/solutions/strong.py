# TIER: strong
# Multi-lattice search + boundary-aware unequal inflation.
#   1. For BOTH a square and a (rotated) hexagonal lattice, binary-search the largest
#      spacing whose N centres nearest the origin still fit inside the arena.
#   2. Take that equal-radius packing (radius s/2), then run an asymmetric radius
#      inflation, seeded from the equal packing, that lets rim / edge zones grow into
#      the wasted arena boundary under wall + neighbour limits. Keep whichever of
#      {equal, inflated} has the larger total (both are feasible).
#   3. Keep whichever lattice yields the larger total radius overall.
# The hex lattice wins on some N, the square lattice on others, so the per-test score
# vector genuinely diverges from the square-only greedy, and both clear the row baseline.
import sys, math


def gen_lattice(R, s, off, kind, rot):
    r = s / 2.0
    lim = R - r
    if lim < 0:
        return []
    c, sn = math.cos(rot), math.sin(rot)
    dy = s * math.sqrt(3.0) / 2.0 if kind == "hex" else s
    n = int(lim / min(s, dy)) + 3
    pts = []
    for j in range(-n, n + 1):
        y = j * dy + off
        xo = (s / 2.0 if (kind == "hex" and (j & 1)) else 0.0) + off
        for i in range(-n, n + 1):
            x = i * s + xo
            X = x * c - y * sn
            Y = x * sn + y * c
            if X * X + Y * Y <= lim * lim + 1e-12:
                pts.append((X, Y))
    pts.sort(key=lambda q: q[0] * q[0] + q[1] * q[1])
    return pts


def best_base(R, N, kind, rots):
    def count(s):
        best = []
        for rot in rots:
            for off in (0.0, s / 2.0):
                p = gen_lattice(R, s, off, kind, rot)
                if len(p) > len(best):
                    best = p
        return best

    lo, hi = 1e-4, 2.0 * R
    bs = lo
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if len(count(mid)) >= N:
            bs = mid
            lo = mid
        else:
            hi = mid
    return count(bs)[:N], bs


def inflate(pts, R, r0, passes=50):
    m = len(pts)
    if m == 0:
        return []
    r = list(r0)
    # After each full pass, every pair (i, j) is bounded by whichever of the two was
    # updated last, so r_i + r_j <= dist_ij holds -> the layout stays non-overlapping.
    for _ in range(passes):
        for i in range(m):
            di = math.hypot(pts[i][0], pts[i][1])
            mn = R - di
            for j in range(m):
                if j == i:
                    continue
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                v = d - r[j]
                if v < mn:
                    mn = v
            r[i] = max(0.0, mn)
    return r


def main():
    t = sys.stdin.read().split()
    N = int(t[0]); R = float(t[1])
    plans = [("square", [0.0]),
             ("hex", [0.0, math.pi / 12.0, math.pi / 6.0])]
    best = None
    for kind, rots in plans:
        pts, s = best_base(R, N, kind, rots)
        if len(pts) < 1:
            continue
        req = s / 2.0
        r_eq = [req] * len(pts)                    # equal packing (feasible)
        r_inf = inflate(pts, R, r_eq)              # asymmetric inflation (feasible)
        r = r_inf if sum(r_inf) >= sum(r_eq) else r_eq
        tot = sum(r)
        if best is None or tot > best[0]:
            best = (tot, pts, r)
    tot, pts, r = best
    lines = [str(len(pts))]
    for k in range(len(pts)):
        lines.append("%.10f %.10f %.10f" % (pts[k][0], pts[k][1], r[k]))
    sys.stdout.write("\n".join(lines) + "\n")


main()
