# TIER: greedy
"""The obvious 'fix' over the floor baseline: round n^D* to the NEAREST
integer cardinality instead of truncating, so the box-counting dimension is
matched much more precisely. Still uses one fixed cardinality at every
level and the same naive row-major (compact/clustered) cell placement --
i.e. it optimizes the named headline metric (dimension) and never looks at
lacunarity at all. This is the trap: on cases where the target lacunarity
is far from what a compact row-major block naturally produces, this
solution is stuck."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); k = int(data[1])
    Dstar = float(data[2])

    N = int(round(n ** Dstar))
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
