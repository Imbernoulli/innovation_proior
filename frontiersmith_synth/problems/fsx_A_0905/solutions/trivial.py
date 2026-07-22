# TIER: trivial
"""
Gradient-blind, zero-adaptivity baseline: color(i,j) = (i+j) mod c.
Moving one step in either row or column direction always changes this
index by exactly 1 (mod c), so no orthogonal run ever exceeds length 1 --
feasible for every K>=1 with no search, no bookkeeping, and no notion of
the target gradient at all. The tile multiset is chosen by the generator
to exactly match this construction's per-color counts, so this always
prints a valid permutation of the required multiset.
This is deliberately identical to the checker's own internal baseline.
"""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); c = int(next(it)); _K = int(next(it))
    for _ in range(c):
        next(it)
    for _ in range(c):
        next(it)
    next(it); next(it)

    out_lines = []
    for i in range(n):
        out_lines.append(" ".join(str((i + j) % c + 1) for j in range(n)))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
