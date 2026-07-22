# TIER: invalid
"""A single self-matching tile type that happily fills the whole window but
always claims value=0 -- wrong for every cell whose target bit is 1. Must
score 0 under the checker's strict feasibility check."""
import sys


def main():
    sys.stdin.read()  # n is irrelevant to this (broken) construction
    # one tile type, glued to itself on every side, value always 0
    print(1)
    print("0 X 2 X 2 X 2 X 2 0")
    print(0)


if __name__ == "__main__":
    main()
