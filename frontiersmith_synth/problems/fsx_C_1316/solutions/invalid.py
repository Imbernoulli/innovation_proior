# TIER: invalid
"""Deliberately infeasible artifact: every position is set to an
out-of-range substituent id (K+999), which violates the [0,K] schema
regardless of the instance's actual K or budget. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    vals = [str(K + 999) for _ in range(N)]
    print(" ".join(vals))


if __name__ == "__main__":
    main()
