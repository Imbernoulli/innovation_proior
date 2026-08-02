# TIER: invalid
"""Deliberately infeasible: emits a full-length rate map that is far outside the
mass budget AND contains a negative rate -- must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[1])
    M = N * N
    vals = [-5.0] + [1.0e9] * (M - 1)
    sys.stdout.write(" ".join("%.4f" % v for v in vals) + "\n")


if __name__ == "__main__":
    main()
