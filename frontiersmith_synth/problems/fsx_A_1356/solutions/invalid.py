# TIER: invalid
"""Emits a garbage, infeasible artifact: m positive numbers that do not sum to
1 (and are not even a distribution), so the checker must score it 0."""
import sys


def main():
    data = sys.stdin.read().split()
    m = int(data[0])
    print(" ".join("2.0" for _ in range(m)))


if __name__ == "__main__":
    main()
