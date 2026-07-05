# TIER: invalid
"""Emits an infeasible artifact: an all-zero matrix (0 is not a valid +/-1
polarity, and it also violates the fixed diagonal), so the checker must reject
it with Ratio 0.0."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    row = " ".join("0" for _ in range(n))
    sys.stdout.write("\n".join(row for _ in range(n)) + "\n")


if __name__ == "__main__":
    main()
