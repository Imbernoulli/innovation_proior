# TIER: invalid
# Emits an out-of-range index -> feasibility breach -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    sys.stdout.write("%d 0 1\n" % (N + 5))


main()
