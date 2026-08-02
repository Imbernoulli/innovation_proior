# TIER: invalid
"""Deliberately infeasible: declares far more tracked buckets than the memory
budget allows (H+G always exceeds M), so the checker must reject it with
Ratio: 0.0 regardless of the trace."""
import sys


def main():
    sys.stdin.read()
    print("500000 500000")


if __name__ == "__main__":
    main()
