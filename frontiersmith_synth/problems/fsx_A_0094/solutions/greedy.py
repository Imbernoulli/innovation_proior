# TIER: greedy
# Spread the stations on the LARGEST inscribed circle (the incircle) instead of
# the small baseline ring. Same concyclic idea, bigger radius -> the thin
# consecutive triple is a constant factor fatter, a clear win over the baseline.
import sys
import math

INR = (2.0 - math.sqrt(2.0)) / 2.0
CX = INR
CY = INR
RINC = INR * 0.999
PHASE = 0.1


def main():
    N = int(sys.stdin.read().split()[0])
    out = []
    for k in range(N):
        ang = 2.0 * math.pi * k / N + PHASE
        out.append("%.10f %.10f" % (CX + RINC * math.cos(ang),
                                    CY + RINC * math.sin(ang)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
