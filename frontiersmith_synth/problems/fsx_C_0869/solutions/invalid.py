# TIER: invalid
"""Emits an out-of-range/garbage artifact: every feature slammed to grade 0 (max
tolerance, min cost) regardless of what the chains require -- infeasible for any
instance with a nontrivial spec bound."""
import sys


def main():
    data = sys.stdin.read().split()
    m = int(data[0])
    sys.stdout.write(" ".join(["0"] * m) + "\n")


if __name__ == "__main__":
    main()
