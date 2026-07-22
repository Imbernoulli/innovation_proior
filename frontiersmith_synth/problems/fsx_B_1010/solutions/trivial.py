# TIER: trivial
"""Floor-rounded single cardinality, row-major (compact) placement, repeated
identically at every level. Ignores lacunarity entirely; even the dimension
fit is sloppy (floor instead of nearest-integer rounding). Reproduces the
checker's own internal baseline construction."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); k = int(data[1])
    Dstar = float(data[2])
    # Lamstar, wD, wL intentionally unused.

    N = int(n ** Dstar)  # floor (int() truncates toward zero for positive floats)
    N = max(1, min(n * n - 1, N))
    order = [(r, c) for r in range(n) for c in range(n)]
    cells = order[:N]

    out = [str(k)]
    for _ in range(k):
        parts = [str(N)]
        for (r, c) in cells:
            parts.append(str(r)); parts.append(str(c))
        out.append(" ".join(parts))
    print("\n".join(out))


if __name__ == "__main__":
    main()
