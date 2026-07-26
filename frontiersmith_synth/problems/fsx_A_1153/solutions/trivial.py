# TIER: trivial
"""Reproduces the checker's own reference construction: consecutive-position
blocks in natural value order. Perfect for row (window) tests that happen to
align with block boundaries, but has no notion of the graft (log-index)
structure at all -- this is the checker's baseline B, so it always scores
~0.1 by definition."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    p = int(nxt())
    k = int(nxt())
    sizes = [int(nxt()) for _ in range(k)]
    # row/graft test parameters are irrelevant to this construction; skip them.

    out = []
    for i in range(1, k + 1):
        out.extend([str(i)] * sizes[i - 1])
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
