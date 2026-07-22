# TIER: invalid
"""Emits a weld order that repeats strut 0 and omits the last strut index --
not a permutation of 0..M-1, must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    for _ in range(m):
        next(it); next(it); next(it)

    out = [str(m)]
    for i in range(m):
        # repeat edge 0 every time instead of a real permutation
        out.append("0 1")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
