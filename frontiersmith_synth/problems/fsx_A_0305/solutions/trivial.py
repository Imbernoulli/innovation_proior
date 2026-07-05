# TIER: trivial
# Reproduces the checker's internal baseline: a smooth centred Gaussian bump.
# By construction c1(f) == B, so the score is ~0.1.
import sys
import math


def main():
    n = int(sys.stdin.read().split()[0])
    if n == 1:
        print("1.0")
        return
    f = [math.exp(-4.0 * ((2.0 * i / (n - 1) - 1.0) ** 2)) for i in range(n)]
    print(" ".join("%.10f" % x for x in f))


if __name__ == "__main__":
    main()
