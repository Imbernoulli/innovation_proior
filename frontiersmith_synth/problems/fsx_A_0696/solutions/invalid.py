# TIER: invalid
"""Deliberately infeasible: dumps every barge into one giant cycle regardless of
direction, capacity, or length limits (and regardless of the reservoir), which
violates the direction-match rule, the chamber count/length caps, or both. Any
single violation must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); C = int(next(it)); L = int(next(it)); t = int(next(it))
    H = int(next(it))
    # ignore the rest of the instance entirely

    idxs = list(range(1, n + 1))
    print(1)
    print("0 0 %d %s" % (len(idxs), " ".join(map(str, idxs))))


if __name__ == "__main__":
    main()
