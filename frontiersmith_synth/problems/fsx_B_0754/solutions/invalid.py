# TIER: invalid
"""Emits an infeasible artifact: dumps an astronomically large purchase on every link,
blowing through the budget constraint by many orders of magnitude. Must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    E = int(toks[0])
    print(" ".join(["1e15"] * E))


if __name__ == "__main__":
    main()
