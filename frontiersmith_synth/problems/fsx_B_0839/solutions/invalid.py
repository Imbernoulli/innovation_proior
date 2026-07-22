# TIER: invalid
"""Garbage plan: references a nonexistent valve id and a negative id.
Must score 0 under strict feasibility validation."""
import sys


def main():
    sys.stdin.read()
    print(3)
    print("0 999999 -1")


if __name__ == "__main__":
    main()
