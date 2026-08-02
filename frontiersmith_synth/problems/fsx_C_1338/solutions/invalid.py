# TIER: invalid
"""Emits a structurally infeasible artifact: only N-1 tokens (wrong count)
AND an out-of-range adhesive index thrown in for good measure. Must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))

    vals = [M + 5] * max(0, N - 1)   # wrong count, and out of range too
    print(" ".join(str(v) for v in vals))


if __name__ == "__main__":
    main()
