# TIER: invalid
"""Emits an obviously infeasible artifact: every slot at every vertex is
labeled with symbol 0, which is not a permutation of {0..m-1} whenever m>1.
Must score 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    for _ in range(n):
        for _ in range(m):
            next(it)
    out = []
    for _v in range(n):
        out.append(" ".join("0" for _ in range(m)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
