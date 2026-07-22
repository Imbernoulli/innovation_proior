# TIER: invalid
"""Deliberately infeasible: every game is dumped onto the same (date,court)
slot, violating both the slot-uniqueness and the one-game-per-team-per-date
constraints. Must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it))
    ngames = N * (N - 1) // 2
    out = ["1 1"] * ngames
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
