# TIER: greedy
# Equal-radius square-grid packing: find the LARGEST lattice spacing s such that
# at least N grid cells fit inside the arena, place the N cells closest to the
# centre, each with radius s/2.  Sum = N * s/2, which beats the row baseline.
import sys, math


def square_pts(R, s, off):
    r = s / 2.0
    lim = R - r
    if lim < 0:
        return []
    n = int(lim / s) + 3
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = i * s + off
            y = j * s + off
            if x * x + y * y <= lim * lim + 1e-12:
                pts.append((x, y))
    pts.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    return pts


def count_best(R, s):
    best = []
    for off in (0.0, s / 2.0):
        p = square_pts(R, s, off)
        if len(p) > len(best):
            best = p
    return best


def main():
    t = sys.stdin.read().split()
    N = int(t[0]); R = float(t[1])
    # binary search for the largest spacing giving at least N cells
    lo, hi = 1e-4, 2.0 * R
    best_s = lo
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if len(count_best(R, mid)) >= N:
            best_s = mid
            lo = mid
        else:
            hi = mid
    pts = count_best(R, best_s)[:N]
    r = best_s / 2.0
    lines = [str(len(pts))]
    for (x, y) in pts:
        lines.append("%.10f %.10f %.10f" % (x, y, r))
    sys.stdout.write("\n".join(lines) + "\n")


main()
