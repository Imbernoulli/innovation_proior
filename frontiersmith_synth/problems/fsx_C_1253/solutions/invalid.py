# TIER: invalid
"""Emits a syntactically-plausible but infeasible plan: Ti=0 is out of the
required range [min_t,max_t] (min_t is always >= 1), so the checker must
reject it with Ratio: 0.0."""
import sys


def main():
    sys.stdin.read()
    print(0, 1, 1)
    print(0, 0, 0)
    print("ijk")


if __name__ == "__main__":
    main()
