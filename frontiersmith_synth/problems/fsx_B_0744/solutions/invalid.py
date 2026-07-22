# TIER: invalid
"""
Deliberately infeasible: emits R non-finite (NaN) tokens. Must score 0
under the checker's strict finiteness gate.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    R = int(toks[0])
    print(" ".join(["nan"] * R))


if __name__ == "__main__":
    main()
