# TIER: invalid
"""Emits a joint angle that violates the stated [-pi, pi] bound on every
waypoint -- must be rejected by the checker (Ratio: 0.0)."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    lines = []
    for _ in range(M):
        row = ["10.0"] + ["0.0"] * (N - 1)
        lines.append(" ".join(row))
    sys.stdout.write("\n".join(lines) + "\n")


main()
