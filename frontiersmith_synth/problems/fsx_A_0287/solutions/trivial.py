# TIER: trivial
# Evenly spaced sensors = an arithmetic progression. For an AP, |A+A| = |A-A|,
# so R = 1 exactly and the score is the calibrated baseline (~0.1).
import sys


def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    d = max(1, M // (n - 1))
    A = [i * d for i in range(n)]
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
