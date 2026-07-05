# TIER: trivial
"""Tune nothing new: echo the givens unchanged. Reproduces the checker baseline
(F == B) -> Ratio ~= 0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    for _ in range(n * n):  # skip district map
        next(it)
    grid = [next(it) for _ in range(n * n)]
    out = []
    k = 0
    for i in range(n):
        row = []
        for j in range(n):
            tok = grid[k]; k += 1
            row.append("." if tok in (".", "-1") else tok)
        out.append(" ".join(row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
