# TIER: invalid
"""Deliberately blows the L1 budget -- must be rejected with Ratio: 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); B = int(toks[1])
    vals = [B + 5] * N
    print(' '.join(str(v) for v in vals))


if __name__ == "__main__":
    main()
