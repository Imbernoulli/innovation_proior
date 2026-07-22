# TIER: trivial
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    cluster_pts = {c: [] for c in range(K)}
    for i in range(1, N + 1):
        x = int(next(it)); y = int(next(it)); c = int(next(it))
        cluster_pts[c].append(i)

    out = []
    out.append(" ".join("0" for _ in range(K)))  # declare every constellation PATH
    for c in range(K):
        pts = cluster_pts[c]
        for i in range(len(pts) - 1):
            out.append(f"{pts[i]} {pts[i + 1]}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
