# TIER: invalid
"""Deliberately infeasible: claims a vertex count that exceeds the stated
cap n, so the checker must reject it outright."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    nH, k, n = map(int, data[0].split())
    N = n + 1000  # violates 1 <= N <= n
    print(N)
    print(0)


if __name__ == "__main__":
    main()
