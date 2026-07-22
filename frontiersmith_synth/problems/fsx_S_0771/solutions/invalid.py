# TIER: invalid
"""Emits an out-of-range member index -> must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    M = int(toks[2])
    print(1)
    print(M + 5)  # out of range [0, M-1]


if __name__ == "__main__":
    main()
