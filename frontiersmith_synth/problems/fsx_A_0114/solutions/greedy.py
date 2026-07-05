# TIER: greedy
# A wider central ring: sweep the ring radius (as a fraction of the inradius) and,
# for each radius, a few rotation phases, keeping the placement with the largest
# minimum-triangle area. Still a single concentric ring inside the incircle, but by
# pushing the modules outward it beats the small baseline ring.
import sys, math


def cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def min_tri(pts):
    n = len(pts); m = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = tri_area(pts[i], pts[j], pts[k])
                if a < m:
                    m = a
    return m


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it))
    A = (float(next(it)), float(next(it)))
    B = (float(next(it)), float(next(it)))
    C = (float(next(it)), float(next(it)))

    a = math.hypot(B[0] - C[0], B[1] - C[1])
    b = math.hypot(C[0] - A[0], C[1] - A[1])
    c = math.hypot(A[0] - B[0], A[1] - B[1])
    per = a + b + c
    ix = (a * A[0] + b * B[0] + c * C[0]) / per
    iy = (a * A[1] + b * B[1] + c * C[1]) / per
    area = 0.5 * abs(cross(A[0], A[1], B[0], B[1], C[0], C[1]))
    r = 2.0 * area / per

    best = None; best_m = -1.0
    for fi in range(1, 20):
        rad = (fi / 20.0) * 0.98 * r
        for pj in range(6):
            ph = pj * (math.pi / 3.0) / 6.0
            ring = [(ix + rad * math.cos(2 * math.pi * i / n + ph),
                     iy + rad * math.sin(2 * math.pi * i / n + ph)) for i in range(n)]
            m = min_tri(ring)
            if m > best_m:
                best_m = m; best = ring

    out = ["%.10f %.10f" % (x, y) for (x, y) in best]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
