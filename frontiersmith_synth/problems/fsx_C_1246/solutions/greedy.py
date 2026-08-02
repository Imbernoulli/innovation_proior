# TIER: greedy
"""The obvious recipe: connect every node to its own M nearest neighbours
(a plain kNN graph), and use the first R point indices as entries. This is
exactly what an average implementation reaches for first -- and it is a
trap: every edge is short, so once the search commits to a starting region
it can never reach a different, far-away cluster; and the first R indices
in the input all sit inside whichever cluster gen.py happened to list
first, so entire other clusters can be structurally unreachable."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it)); R = int(next(it))
    pts = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it))
        pts.append((x, y))
    Q = int(next(it))
    for _ in range(Q):
        next(it); next(it)

    def d2(a, b):
        dx = a[0] - b[0]; dy = a[1] - b[1]
        return dx * dx + dy * dy

    out = []
    entries = list(range(R))
    out.append(" ".join(str(e) for e in entries))
    for i in range(N):
        dists = sorted((d2(pts[i], pts[j]), j) for j in range(N) if j != i)
        nbrs = [j for _, j in dists[:M]]
        out.append(str(len(nbrs)) + " " + " ".join(str(v) for v in nbrs))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
