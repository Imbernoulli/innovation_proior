# TIER: trivial
"""Simplest legal plan: the smallest allowed tile, zero padding, canonical
i,j,k inner order. This is EXACTLY the checker's internal baseline
construction B, so it always scores ~0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    min_t = min(2, max(1, N // 3))
    print(min_t, min_t, min_t)
    print(0, 0, 0)
    print("ijk")


if __name__ == "__main__":
    main()
