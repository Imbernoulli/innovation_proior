# TIER: invalid
"""Deliberately infeasible: a single self-sticky tile type (every side
shares the same strength-2 glue) happily binds to itself in all four
directions, so it grows into an ever-expanding diamond instead of the
required k x (T+1) rectangle -- it never terminates inside the shape and
never matches the target, so the checker must reject it with Ratio 0.
"""
import sys


def main():
    sys.stdin.read()
    print(1)
    print("7 2 7 2 7 2 7 2")
    print(1)


if __name__ == "__main__":
    main()
