# TIER: trivial
"""Trivial baseline: evenly spaced tide pools (arithmetic progression).

This reproduces the checker's internal baseline A0 exactly, so |A+A| = |A-A| and
R = 1 -> Ratio = 0.1. It is the naive 'sample the shelf uniformly' layout."""
import sys


def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    d = max(1, M // (n - 1))
    A = [i * d for i in range(n)]
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
