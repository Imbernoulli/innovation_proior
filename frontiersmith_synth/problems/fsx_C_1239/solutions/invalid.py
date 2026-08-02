# TIER: invalid
"""Emits an out-of-range artifact: one too few tokens AND a group id that is
out of bounds, so the checker's strict feasibility gate must reject it with
Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    M = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1

    # Deliberately infeasible: only M-1 tokens, and the one that IS group K
    # (one past the valid [0,K-1] range) even if the count were tolerated.
    out = [str(K)] + ["0"] * (M - 2)
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
