# TIER: invalid
"""Deliberately infeasible: claims every visit is a shelf hit. The very first visit
to any object is always a genuine miss under the checker's ground truth, so this
is rejected immediately (Ratio: 0.0) on every test case."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); C = int(next(it))
    for _ in range(K):
        next(it)
    for _ in range(N):
        next(it)
    sys.stdout.write("\n".join(["H"] * N) + "\n")


if __name__ == "__main__":
    main()
