# TIER: invalid
"""Emits a garbage / infeasible artifact: an out-of-grammar expression that
references an undeclared variable and uses a disallowed operator -- the
checker must reject it and score 0."""
import sys


def main():
    sys.stdin.read()
    print("q ** 2 + banana(p, L)")


if __name__ == "__main__":
    main()
