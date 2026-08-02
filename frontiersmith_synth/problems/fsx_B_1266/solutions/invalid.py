# TIER: invalid
# Deliberately infeasible: dumps all capital into sleeve 0 (ignoring both its individual
# cap and, if sleeve 0 is in the cluster, the group cap), and puts nothing anywhere else --
# the checker must reject this on the cap check (or the sum-to-1 check for N=1 edge cases
# it still ignores caps) and print Ratio: 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    pos = 0
    tid = int(toks[pos]); pos += 1
    N = int(toks[pos]); pos += 1
    w = [0.0] * N
    w[0] = 1.0
    sys.stdout.write(" ".join("%.8f" % v for v in w) + "\n")


main()
