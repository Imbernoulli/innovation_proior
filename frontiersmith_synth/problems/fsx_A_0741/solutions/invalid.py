# TIER: invalid
"""Emits a syntactically-plausible but infeasible merge plan: a range with
first > last (never valid, regardless of the instance), so the checker must
reject it and score 0 on every test."""
import sys


def main():
    sys.stdin.read()
    print(3)
    print("0 5 2")
    print("1000000 1 1")
    print("0 1 1")


if __name__ == "__main__":
    main()
