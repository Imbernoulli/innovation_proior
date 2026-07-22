# TIER: invalid
"""Deliberately infeasible: dumps the budget onto a single segment and leaves
every other segment at stiffness 0 (violates k_j >= 1), and the printed sum
does not equal BUDGET either (off by a garbage additive fudge). Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    S = int(data[0]); BUDGET = int(data[1])
    ks = [0] * S
    ks[0] = BUDGET + 12345  # neither respects the per-segment minimum nor the exact budget
    print(" ".join(map(str, ks)))


if __name__ == "__main__":
    main()
