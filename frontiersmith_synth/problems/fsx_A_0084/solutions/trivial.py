# TIER: trivial
# Baseline: N sensors equally spaced on the triangle's incircle
# (the same construction the checker uses as its reference baseline).
import sys, math


def main():
    t = sys.stdin.read().split()
    N = int(t[0])
    A = (float(t[1]), float(t[2]))
    B = (float(t[3]), float(t[4]))
    C = (float(t[5]), float(t[6]))
    a = math.hypot(B[0] - C[0], B[1] - C[1])
    b = math.hypot(C[0] - A[0], C[1] - A[1])
    c = math.hypot(A[0] - B[0], A[1] - B[1])
    s = 0.5 * (a + b + c)
    area = math.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c)))
    r = 0.6 * area / s  # small ring == checker's baseline construction
    ix = (a * A[0] + b * B[0] + c * C[0]) / (a + b + c)
    iy = (a * A[1] + b * B[1] + c * C[1]) / (a + b + c)
    out = []
    for k in range(N):
        ang = 2.0 * math.pi * k / N
        out.append("%.12f %.12f" % (ix + r * math.cos(ang), iy + r * math.sin(ang)))
    sys.stdout.write("\n".join(out) + "\n")


main()
