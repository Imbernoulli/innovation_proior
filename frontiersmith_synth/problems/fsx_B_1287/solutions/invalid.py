# TIER: invalid
"""Emits an out-of-bound garbage hedge sequence -- must score 0 under the feasibility gate."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    print(" ".join("1e9" for _ in range(N)))


if __name__ == "__main__":
    main()
