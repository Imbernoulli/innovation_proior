# TIER: invalid
"""Emits an out-of-range reaction index -- must score 0."""
import sys


def main():
    tokens = sys.stdin.read().split()
    m = int(tokens[1])
    print(1)
    print(m)  # valid indices are 0..m-1; m itself is out of range


if __name__ == "__main__":
    main()
