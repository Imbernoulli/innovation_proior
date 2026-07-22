# TIER: strong
# The insight: a feedback exponent mu > 1 makes flux that is SHARED by several
# terminal pairs on the same edge reinforce super-additively (since flux from
# every pair is summed before the ^mu nonlinearity is applied), so an edge
# lying near the geometric median of the terminals -- reachable by none of the
# pairwise shortest paths individually, but touched a little by all of them --
# can be grown into a genuine Steiner junction that a linear (mu=1) recipe
# never discovers. We seed the pairwise shortest paths at only HALF strength
# (so the dynamics stay free to abandon redundant rim segments) plus a small
# extra nudge at the terminals' geometric median (Weiszfeld iteration, snapped
# to the nearest free cell) to break the initial symmetry toward that hub, and
# hand the merging exponent mu=2.3 to the checker's dynamics to do the rest.
import sys
import math
from collections import deque


def build_grid(R, C, obstacles, terminals):
    node_id = {}
    coords = []
    for r in range(R):
        for c in range(C):
            if (r, c) in obstacles:
                continue
            node_id[(r, c)] = len(coords)
            coords.append((r, c))
    edges = []
    for r in range(R):
        for c in range(C):
            if (r, c) in obstacles:
                continue
            u = node_id[(r, c)]
            if (r, c + 1) in node_id:
                edges.append((u, node_id[(r, c + 1)]))
            if (r + 1, c) in node_id:
                edges.append((u, node_id[(r + 1, c)]))
    term_nodes = [node_id[t] for t in terminals]
    return node_id, coords, edges, term_nodes


def bfs_path(n, edges, s, t):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    prev = [-1] * n
    seen = [False] * n
    seen[s] = True
    q = deque([s])
    while q:
        x = q.popleft()
        if x == t:
            break
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True
                prev[y] = x
                q.append(y)
    path = []
    x = t
    while x != -1 and x != s:
        p = prev[x]
        if p == -1:
            return []
        path.append((p, x))
        x = p
    path.reverse()
    return path


def geometric_median(pts, iters=50):
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    for _ in range(iters):
        num_x = num_y = den = 0.0
        for (px, py) in pts:
            d = math.hypot(px - x, py - y) + 1e-6
            num_x += px / d
            num_y += py / d
            den += 1.0 / d
        x, y = num_x / den, num_y / den
    return x, y


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    R = int(nxt()); C = int(nxt()); K = int(nxt())
    terminals = [(int(nxt()), int(nxt())) for _ in range(K)]
    W = int(nxt())
    obstacles = set((int(nxt()), int(nxt())) for _ in range(W))

    node_id, coords, edges, term_nodes = build_grid(R, C, obstacles, terminals)
    n = len(coords)
    edge_index = {}
    for (u, v) in edges:
        edge_index[(coords[u], coords[v])] = True
        edge_index[(coords[v], coords[u])] = True

    D_MAX = 5.0
    PATH_FRAC = 0.5
    HUB_BOOST = 2.5
    MU = 2.3

    overrides = {}
    for i in range(K):
        for j in range(i + 1, K):
            path = bfs_path(n, edges, term_nodes[i], term_nodes[j])
            for (u, v) in path:
                overrides[(coords[u], coords[v])] = D_MAX * PATH_FRAC

    gx, gy = geometric_median(terminals)
    best, bestd = None, float("inf")
    for (r, c) in node_id:
        dd = (r - gx) ** 2 + (c - gy) ** 2
        if dd < bestd:
            bestd, best = dd, (r, c)
    br, bc = best
    for (dr, dc) in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nb = (br + dr, bc + dc)
        if nb in node_id:
            overrides[((br, bc), nb)] = HUB_BOOST

    lines = [str(MU), str(len(overrides))]
    for (a, b), d0 in overrides.items():
        lines.append("%d %d %d %d %.4f" % (a[0], a[1], b[0], b[1], d0))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
