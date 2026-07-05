# TIER: invalid
"""Emits an infeasible artifact: a MAP that is NOT a permutation (all robots on
bay 0).  The checker's feasibility gate must reject it -> Ratio 0.0."""
import sys


def main():
    tok = sys.stdin.read().split()
    V = int(tok[0])
    sys.stdout.write("MAP " + " ".join("0" for _ in range(V)) + "\n")


if __name__ == "__main__":
    main()
