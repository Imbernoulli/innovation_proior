# TIER: invalid
"""Deliberately infeasible artifact: claims an edge index far outside the netlist's range.
Must score 0.0 under strict feasibility checking."""
import sys


def main():
    sys.stdin.read()
    print(1)
    print(999999, 42.0)


if __name__ == "__main__":
    main()
