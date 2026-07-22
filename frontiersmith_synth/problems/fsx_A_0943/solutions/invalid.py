# TIER: invalid
"""Deliberately infeasible: connects the first star of constellation 0 to the first star
of constellation 1 (a cross-constellation edge), which the checker must reject."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    cluster_pts = {c: [] for c in range(K)}
    for i in range(1, N + 1):
        x = int(next(it)); y = int(next(it)); c = int(next(it))
        cluster_pts[c].append(i)

    out = [" ".join("1" for _ in range(K))]  # declare everything STAR, doesn't matter
    # emit a valid-looking chain for cluster 0 but then splice in a cross-cluster edge
    # and drop one legitimate edge, so both the "own cluster" and the tree-count checks
    # are violated.
    for c in range(K):
        pts = cluster_pts[c]
        for i in range(len(pts) - 1):
            if c == 0 and i == 0 and K > 1 and cluster_pts[1]:
                out.append(f"{pts[0]} {cluster_pts[1][0]}")  # illegal cross-cluster edge
            else:
                out.append(f"{pts[i]} {pts[i + 1]}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
