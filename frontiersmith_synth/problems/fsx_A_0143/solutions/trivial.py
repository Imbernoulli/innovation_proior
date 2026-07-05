# TIER: trivial
# Reproduces the checker baseline: N equal zones in a single row along the diameter,
# each of radius R/N. Sum of radii = R = B  ->  Ratio ~ 0.1.
import sys


def main():
    t = sys.stdin.read().split()
    N = int(t[0]); R = float(t[1])
    rho = R / N
    lines = [str(N)]
    for k in range(N):
        x = (2 * k - (N - 1)) * rho
        lines.append("%.10f %.10f %.10f" % (x, 0.0, rho))
    sys.stdout.write("\n".join(lines) + "\n")


main()
