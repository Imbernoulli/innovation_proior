# TIER: invalid
"""Emits a single layer whose material index is K+3 -- out of the valid
[1,K] range. The checker must reject this (malformed output) -> 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    K = int(toks[1])
    print(1)
    print("%d %.3f" % (K + 3, 120.0))


if __name__ == "__main__":
    main()
