# TIER: invalid
"""Deliberately infeasible: claims the FULL grid (N = n*n) as its cardinality
at every level, which exceeds the required N <= n*n-1 (a full grid is not a
proper fractal -- no gaps at all) and also duplicates a cell. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); k = int(data[1])

    out = [str(k)]
    for _ in range(k):
        N = n * n
        parts = [str(N)]
        cells = [(r, c) for r in range(n) for c in range(n)]
        cells.append(cells[0])  # duplicate -> also invalid on its own
        for (r, c) in cells:
            parts.append(str(r)); parts.append(str(c))
        out.append(" ".join(parts))
    print("\n".join(out))


if __name__ == "__main__":
    main()
