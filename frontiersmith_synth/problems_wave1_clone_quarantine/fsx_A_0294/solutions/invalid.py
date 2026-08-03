# TIER: invalid
# Emits positions OUTSIDE the hall -> feasibility gate must score 0.
import sys


def main():
    t = sys.stdin.read().split()
    N = int(t[0])
    out = []
    for k in range(N):
        out.append("%.6f %.6f" % (5.0 + k, -3.0 - k))
    sys.stdout.write("\n".join(out) + "\n")


main()
