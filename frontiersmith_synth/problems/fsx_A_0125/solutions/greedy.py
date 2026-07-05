# TIER: greedy
# Spread the gain uniformly across the whole array -- the obvious way to avoid
# dumping leakage into a few bands. Roughly halves the L2 leakage energy versus
# the half-block baseline (~0.20 ratio).
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    f = [1.0] * n
    print(" ".join("%.6f" % x for x in f))


if __name__ == "__main__":
    main()
