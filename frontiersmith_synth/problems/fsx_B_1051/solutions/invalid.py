# TIER: invalid
"""Emits a clearly infeasible artifact: the same clan repeated n times (not a
permutation at all). Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    n1, n2 = int(data[0]), int(data[1])
    n = n1 * n2
    out = "\n".join("0 0" for _ in range(n))
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
