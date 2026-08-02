# TIER: trivial
"""Ignores geometry entirely: entries = first R indices, and every node's
out-edges are the next `deg` indices cyclically after it in input order.
This reproduces the judge's own "no index" baseline in spirit -- it neither
looks at distances nor at cluster structure, so it succeeds only by
accident (and, like the obvious kNN recipe, stays entirely index-local, so
it has no long-range reach either)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it)); R = int(next(it))
    for _ in range(N):
        next(it); next(it)
    Q = int(next(it))
    for _ in range(Q):
        next(it); next(it)

    out = []
    entries = list(range(R))
    out.append(" ".join(str(e) for e in entries))
    for i in range(N):
        deg = min(M, N - 1)
        nbrs = [(i + k) % N for k in range(1, deg + 1)]
        out.append(str(deg) + " " + " ".join(str(v) for v in nbrs))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
