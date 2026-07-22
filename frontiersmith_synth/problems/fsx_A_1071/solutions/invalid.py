# TIER: invalid
"""Deliberately infeasible: every row is loaded with ALL n references (row
weight n, not k), and one entry is pushed out of range. Must score 0."""
import sys


def main():
    n, k = map(int, sys.stdin.read().split()[:2])
    for i in range(n):
        row = [1] * n
        row[0] = 2  # out of {-1,0,1} range
        print(" ".join(map(str, row)))


if __name__ == "__main__":
    main()
