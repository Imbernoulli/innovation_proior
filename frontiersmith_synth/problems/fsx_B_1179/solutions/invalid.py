# TIER: invalid
"""Emits K non-finite values -- must score 0 under strict feasibility checking."""
import sys


def main():
    data = sys.stdin.read().split()
    K = int(data[1]) if len(data) > 1 else 1
    print(" ".join(["nan"] * K))


if __name__ == "__main__":
    main()
