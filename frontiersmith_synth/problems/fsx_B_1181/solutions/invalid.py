# TIER: invalid
"""Emits an out-of-range artifact (negative R0) -- must score 0."""
import sys


def main():
    sys.stdin.read()
    print("-1.0 1 0.0")
    print("0.05 100.0")


if __name__ == "__main__":
    main()
