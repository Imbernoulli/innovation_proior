# TIER: invalid
"""Deliberately infeasible: floods the grid with color 1 everywhere, which both
breaks the fixed tile multiset (unless c==1) and blows the run-length cap on
every row and column. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); c = int(next(it)); K = int(next(it))
    for _ in range(c):
        next(it)
    for _ in range(c):
        next(it)
    next(it); next(it)

    out_lines = []
    for _ in range(n):
        out_lines.append(" ".join(["1"] * n))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
