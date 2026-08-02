# TIER: invalid
"""Emits a garbage mask: wrong dimensions AND invalid characters, so the
checker's feasibility gate must reject it with Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # too few rows, wrong width, and non-{0,1} characters
    for _ in range(max(1, n - 3)):
        print("2" * (n + 5))


if __name__ == "__main__":
    main()
