# TIER: trivial
# Baseline: N towers equally spaced on a circle of radius 0.20 about the plot
# center (the same construction the checker uses as its reference baseline).
import sys, math

RING_R = 0.20
CX, CY = 0.5, 0.5


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
