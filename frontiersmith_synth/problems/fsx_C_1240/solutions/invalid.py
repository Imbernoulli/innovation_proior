# TIER: invalid
"""Deliberately infeasible artifact: wrong token count AND out-of-range
shard indices. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    K = int(data[1])
    # emit n-1 tokens (wrong count) and push every value out of [0,K)
    vals = [str(K + 5) for _ in range(max(0, n - 1))]
    print(" ".join(vals))


if __name__ == "__main__":
    main()
