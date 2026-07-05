# TIER: trivial
# Reproduces the checker's internal baseline: N stations equally spaced on a
# small circle of radius r_in/sqrt(3) about the incenter -> Ratio ~= 0.1.
import sys
import math

INR = (2.0 - math.sqrt(2.0)) / 2.0
CX = INR
CY = INR
RINC = INR * 0.999
R_BASE = RINC / math.sqrt(3.0)
PHASE = 0.1


def main():
    N = int(sys.stdin.read().split()[0])
    out = []
    for k in range(N):
        ang = 2.0 * math.pi * k / N + PHASE
        out.append("%.10f %.10f" % (CX + R_BASE * math.cos(ang),
                                    CY + R_BASE * math.sin(ang)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
