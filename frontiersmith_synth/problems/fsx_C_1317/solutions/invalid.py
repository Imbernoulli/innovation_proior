# TIER: invalid
"""Deliberately infeasible: dumps a monomer type index that is out of range
for the instance (K+5), which must be rejected by the checker."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    seq = [K + 5] * N
    print(" ".join(map(str, seq)))


if __name__ == "__main__":
    main()


