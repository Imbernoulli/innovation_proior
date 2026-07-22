# TIER: invalid
"""Emits an out-of-range garbage generator -> must score 0 (feasibility gate)."""
import sys


def main():
    sys.stdin.read()
    print("1e30 1e30 1e30 1e30 1e30 1e30")


if __name__ == "__main__":
    main()
