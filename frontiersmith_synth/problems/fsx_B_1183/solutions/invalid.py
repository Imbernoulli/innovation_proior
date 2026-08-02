# TIER: invalid
"""Deliberately garbage / infeasible artifact (non-finite values)."""
import sys


def main():
    sys.stdin.read()
    print("nan nan nan nan")


if __name__ == "__main__":
    main()
