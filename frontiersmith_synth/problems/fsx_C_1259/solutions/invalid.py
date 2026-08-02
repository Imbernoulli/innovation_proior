# TIER: invalid
"""Deliberately infeasible: wrong token count AND out-of-range values."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); g = int(next(it))
    print(" ".join(["2"] * (g + 1)))


if __name__ == "__main__":
    main()
