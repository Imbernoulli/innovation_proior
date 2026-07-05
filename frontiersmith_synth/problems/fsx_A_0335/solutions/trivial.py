# TIER: trivial
# Reproduces the checker's naive boundary-loaded baseline -> Ratio ~= 0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = []
    for i in range(n):
        x = (i + 0.5) / n
        out.append(1.0 + 8.0 * ((2.0 * x - 1.0) ** 4))
    print(" ".join("%.10g" % v for v in out))


if __name__ == "__main__":
    main()
