# TIER: trivial
# Reproduces the checker's internal baseline: a single concentrated coolant
# block of width max(1, n//5). Scores Ratio ~ 0.1 by construction.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    w = max(1, n // 5)
    f = [1] * w + [0] * (n - w)
    sys.stdout.write(" ".join(map(str, f)) + "\n")


if __name__ == "__main__":
    main()
