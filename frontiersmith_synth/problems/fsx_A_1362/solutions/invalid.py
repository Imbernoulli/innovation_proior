# TIER: invalid
"""Deliberately infeasible garbage: negative fractions that also don't sum to 1.
Must score 0 on every instance regardless of its valuations."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    row = " ".join("-0.5" for _ in range(n))
    sys.stdout.write("\n".join(row for _ in range(m)) + "\n")


if __name__ == "__main__":
    main()
