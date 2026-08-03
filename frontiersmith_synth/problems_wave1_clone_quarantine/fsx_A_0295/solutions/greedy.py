# TIER: greedy
# Uniform capacity across the whole chain. A perfectly flat allocation already
# spreads the stress spectrum and roughly halves the L2 energy versus the
# half-block baseline -- but it is far from the tapered optimum.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    f = [1.0] * n
    print(" ".join("%.6f" % x for x in f))


if __name__ == "__main__":
    main()
