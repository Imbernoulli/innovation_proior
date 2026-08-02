# TIER: invalid
"""Emits an out-of-range parameter (k_trip > w_trip) -- must score 0."""
import sys


def main():
    sys.stdin.readline()
    sys.stdin.readline()
    print("3 999 5 3 4 1 2")


if __name__ == "__main__":
    main()
