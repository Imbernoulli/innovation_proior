# TIER: trivial
# Baseline: N pads equally spaced on a circle of radius 0.15 about the plate
# centroid (the same construction the checker uses as its reference baseline).
import sys, math

RING_R = 0.15
CX, CY = 1.0 / 3.0, 1.0 / 3.0


def main():
    t = sys.stdin.read().split()
    N = int(t[0])
    out = []
    for k in range(N):
        ang = 2.0 * math.pi * k / N
        out.append("%.12f %.12f" % (CX + RING_R * math.cos(ang),
                                    CY + RING_R * math.sin(ang)))
    sys.stdout.write("\n".join(out) + "\n")


main()
