# TIER: trivial
# Reproduces the checker baseline: a regular n-gon of radius 0.55*inradius centred at
# the plot's incentre (a small central ring). Min triangle area = B  ->  Ratio = 0.1
import sys, math


def cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


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
    rad = 0.55 * r

    out = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        out.append("%.10f %.10f" % (ix + rad * math.cos(th), iy + rad * math.sin(th)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
