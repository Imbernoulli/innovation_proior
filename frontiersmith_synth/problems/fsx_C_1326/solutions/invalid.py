# TIER: invalid
"""Emits a structurally infeasible formulation: solvent fractions that do not
sum to 1 (they sum to roughly N, wildly over-allocated) and an additive
loading vector that blows straight through the shared budget. Must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))

    x = [1.0] * N          # sums to N, not 1 -> infeasible
    a = [1000.0] * M       # way over any A_max -> infeasible

    print(" ".join(f"{v:.6f}" for v in x))
    print(" ".join(f"{v:.6f}" for v in a))


if __name__ == "__main__":
    main()
