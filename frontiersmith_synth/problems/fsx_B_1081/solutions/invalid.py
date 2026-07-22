# TIER: invalid
"""Deliberately infeasible: claims an out-of-range panel id (N, one past the
valid [0,N-1] range) for every triangle. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    R = int(data[0]); C = int(data[1])
    N = 2 * R * C
    sys.stdout.write(" ".join([str(N)] * N) + "\n")


if __name__ == "__main__":
    main()
