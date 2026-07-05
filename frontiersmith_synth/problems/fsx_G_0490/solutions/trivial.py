# TIER: trivial
"""Reproduce the checker's cyclic baseline exactly -> F == B -> Ratio ~ 0.1."""
import sys
from math import gcd


def build_baseline(n, k):
    coprimes = [a for a in range(1, n) if gcd(a, n) == 1]
    L = len(coprimes)
    squares = []
    for m in range(k):
        a = coprimes[m % L]
        s = m // L
        sq = [[(a * i + j + s) % n for j in range(n)] for i in range(n)]
        squares.append(sq)
    return squares


def main():
    data = sys.stdin.read().split()
    n, k = int(data[0]), int(data[1])
    squares = build_baseline(n, k)
    out = []
    for sq in squares:
        for row in sq:
            out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
