# TIER: trivial
# Reproduces the checker's baseline: the "half-block" allocation.
# All gain crammed into the first floor(n/2) sensors -> concentrated leakage
# spectrum -> ratio ~= 0.1.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    f = [1.0] * (n // 2) + [0.0] * (n - n // 2)
    print(" ".join("%.6f" % x for x in f))


if __name__ == "__main__":
    main()
