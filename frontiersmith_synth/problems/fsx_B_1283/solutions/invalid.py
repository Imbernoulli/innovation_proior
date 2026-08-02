# TIER: invalid
"""Emits an infeasible roster: one block's senior headcount exceeds
MAX_PER_SLOT (read straight from the input, then blown past), so the
checker's feasibility gate must reject it -> Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    T = int(data[p]); p += 1
    n_starts = int(data[p]); p += 1
    p += n_starts
    max_per_slot = int(data[p]); p += 1
    for b in range(n_starts):
        if b == 0:
            print(3, max_per_slot + 5)  # out-of-range: violates 0<=count<=MAX_PER_SLOT
        else:
            print(2, 2)


if __name__ == "__main__":
    main()
