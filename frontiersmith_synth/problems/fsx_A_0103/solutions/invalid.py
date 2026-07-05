# TIER: invalid
"""Infeasible artifact: every dome stacked on the antenna, mutually overlapping and
straddling the keep-out. Must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    for _ in range(N):
        print("0.5 0.5 0.4")


if __name__ == "__main__":
    main()
