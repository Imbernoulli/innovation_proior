# TIER: greedy
"""The "obvious" textbook approach: build the Euclidean minimum spanning tree inside
every constellation independently (self-crossing-free, locally near-optimal), then
guess a role from the tree it happened to produce. It never looks at OTHER
constellations, so on a batch of similarly-distributed clusters it produces near-
identical shapes and near-identical role guesses everywhere -- no portfolio diversity."""
import sys, math


def mst_edges(pts, pos):
    s = len(pts)
    in_tree = [False] * s
    dist = [float("inf")] * s
    parent = [-1] * s
    dist[0] = 0
    edges = []
    for _ in range(s):
        u = -1
        best = float("inf")
        for i in range(s):
            if not in_tree[i] and dist[i] < best:
                best = dist[i]; u = i
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((pts[parent[u]], pts[u]))
        ux, uy = pos[pts[u]]
        for v in range(s):
            if not in_tree[v]:
                vx, vy = pos[pts[v]]
                d = (ux - vx) ** 2 + (uy - vy) ** 2
                if d < dist[v]:
                    dist[v] = d; parent[v] = u
    return edges


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    pos = {}
    cluster_pts = {c: [] for c in range(K)}
    for i in range(1, N + 1):
        x = int(next(it)); y = int(next(it)); c = int(next(it))
        pos[i] = (x, y)
        cluster_pts[c].append(i)

    roles = []
    all_edges = []
    for c in range(K):
        pts = cluster_pts[c]
        edges = mst_edges(pts, pos)
        all_edges.append(edges)
        deg = {p: 0 for p in pts}
        for (u, v) in edges:
            deg[u] += 1; deg[v] += 1
        max_deg = max(deg.values()) if deg else 1
        # a "locally reasonable" role guess based on this tree alone
        if max_deg >= 4:
            roles.append(1)  # looks branchy -> STAR
        else:
            roles.append(0)  # looks path-ish -> PATH (MST rarely looks zigzag)

    out = [" ".join(str(r) for r in roles)]
    for edges in all_edges:
        for (u, v) in edges:
            out.append(f"{u} {v}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
