# TIER: trivial
# Baseline: N attendees equally spaced on the radius-0.3 concentric ring about
# the hall centre (the same construction the checker uses as its reference
# baseline). Scores ~0.1 by design.
import sys, math

CX, CY, RBASE = 0.5, 0.5, 0.3


def main():
    t = sys.stdin.read().split()
    N = int(t[0])
    out = []
    for k in range(N):
        ang = 2.0 * math.pi * k / N
        out.append("%.12f %.12f" % (CX + RBASE * math.cos(ang),
                                    CY + RBASE * math.sin(ang)))
    sys.stdout.write("\n".join(out) + "\n")


main()
