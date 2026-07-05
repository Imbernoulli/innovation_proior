# TIER: invalid
"""Infeasible: a single all-zero rank-1 term reconstructs the zero tensor, which does
NOT equal the (nonzero) target -> checker must score 0."""
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    print(1)
    print(" ".join(["0"] * (a + b + c)))


if __name__ == "__main__":
    main()
