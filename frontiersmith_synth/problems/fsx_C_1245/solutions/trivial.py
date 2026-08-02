# TIER: trivial
"""Reproduces the checker's own baseline construction exactly: one block
covering the whole stream at its own minimal feasible width. No partitioning,
no palette choice -- this is 'do nothing clever'."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); H = int(next(it)); C = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    base = min(A)
    d = max(A) - base
    w = d.bit_length()

    print(1)
    print(N, w)


if __name__ == "__main__":
    main()
