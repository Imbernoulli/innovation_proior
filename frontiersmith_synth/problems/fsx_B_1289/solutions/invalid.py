# TIER: invalid
"""Deliberately infeasible artifact: claims a single stage but sets its
boundary far beyond M (and, redundantly, would also violate g_K == M even if
M happened to be that large). Must score 0 under strict feasibility checks."""
import sys


def main():
    sys.stdin.read()  # ignore the instance entirely
    print(1)
    print(999999999)


if __name__ == "__main__":
    main()
