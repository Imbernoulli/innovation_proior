# TIER: invalid
"""Emits an infeasible artifact (out-of-alphabet glyphs) -- must score 0."""
import sys


def main():
    sys.stdin.read()
    print("0 XY Z")
    print("1 AB Q")


if __name__ == "__main__":
    main()
