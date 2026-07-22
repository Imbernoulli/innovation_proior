# TIER: invalid
# Infeasible artifact: every organ points to a non-existent parent node, so the
# checker's parent-range / tree validation rejects it -> Ratio 0.0 on every case.
import sys


def main():
    t = sys.stdin.read().split()
    K = int(t[0])
    out = ["0"]                       # no junctions
    for _ in range(K):
        out.append(str(10 ** 9))      # parent index far out of range
    sys.stdout.write("\n".join(out) + "\n")


main()
