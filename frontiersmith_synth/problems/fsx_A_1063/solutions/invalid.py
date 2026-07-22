# TIER: invalid
"""Deliberately garbage / infeasible output: violates the budget constraint
(and, for good measure, emits an out-of-range value) so the checker must
score this Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    n = int(data[p]); p += 1
    # ignore the rest of the instance entirely

    e = [10 ** 6] * n  # wildly out of range AND breaks the budget
    print(" ".join(map(str, e)))


if __name__ == "__main__":
    main()
