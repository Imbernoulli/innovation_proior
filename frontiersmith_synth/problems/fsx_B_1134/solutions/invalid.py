# TIER: invalid
"""Emits out-of-range garbage angles -- must be rejected (Ratio: 0.0)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    K = int(next(it))
    n = 4 * K + (K - 1)
    print(" ".join("7.777" for _ in range(n)))


if __name__ == "__main__":
    main()
