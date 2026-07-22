# TIER: invalid
"""Garbage artifact: every symbol assigned to slot 0 -- not a bijection, must
score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    print(" ".join("0" for _ in range(n)))


if __name__ == "__main__":
    main()
