# TIER: invalid
"""
Emits a syntactically-allowed expression that is infeasible at every
held-out batch: division by zero (x - x == 0), which the checker must
reject as a non-finite prediction -> Ratio 0.
"""
import sys


def main():
    sys.stdin.read()
    print("x / (x - x)")


if __name__ == "__main__":
    main()
