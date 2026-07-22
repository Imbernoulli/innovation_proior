# TIER: greedy
# The textbook recipe: for EVERY pair of terminals, compute a shortest path
# through the maze (BFS) and reinforce it maximally (mu=1.0, no synergy
# exponent). This is precisely "reinforcing every currently-shortest pairwise
# path" -- it produces a feasible, obstacle-aware network, but the union of
# many separately-chosen pairwise paths overlaps redundantly near the rim
# instead of merging into a cheaper interior junction, so it lands well above
# the true Steiner-optimal tube count.
import sys
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
        edge_index[(coords[u], coords[v])] = 1
        edge_index[(coords[v], coords[u])] = 1

    D_MAX = 5.0
    overrides = {}
    for i in range(K):
        for j in range(i + 1, K):
            path = bfs_path(n, edges, term_nodes[i], term_nodes[j])
            for (u, v) in path:
                overrides[(coords[u], coords[v])] = D_MAX

    lines = ["1.0", str(len(overrides))]
    for (a, b), d0 in overrides.items():
        lines.append("%d %d %d %d %.4f" % (a[0], a[1], b[0], b[1], d0))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
