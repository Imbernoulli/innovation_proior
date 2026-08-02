# TIER: invalid
"""Activates every position -- always exceeds the budget B < L, so the
checker must reject it before scoring (score 0)."""
import sys


def main():
    data = sys.stdin.read().split()
    L = int(data[0])
    sys.stdout.write(" ".join("1" for _ in range(L)) + "\n")


if __name__ == "__main__":
    main()
