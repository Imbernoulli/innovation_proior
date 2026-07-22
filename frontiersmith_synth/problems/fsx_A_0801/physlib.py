"""Shared physarum-maze core for fsx_A_0801: grid/graph construction, BFS, the
current-injection electrical solve, and the reinforcement dynamics. Imported by
gen.py and verify.py only (solutions are sandboxed standalone and inline their
own copies of the small BFS helper). Pure functions; no randomness, no wall
clock -- fully deterministic given the same D0/mu inputs.
"""
import math
from collections import deque

D_MIN = 0.02
D_BASE = 0.30
D_MAX = 5.0
MU_MIN = 0.2
MU_MAX = 4.0
ALPHA = 0.4
ROUNDS = 8
THRESH = 0.12
MAX_OVERRIDES = 4000


def build_grid(R, C, obstacles, terminals):
    """obstacles: set of (r,c). terminals: list of (r,c) in fixed input order.
    Returns node_id (dict (r,c)->int, row-major insertion order), coords (list
    of (r,c) indexed by node id), edges (list of (u,v) int node-id pairs, in
    canonical row-major (right-then-down) order), term_nodes (list of int)."""
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


def adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def bfs_reachable_all(n, edges, start, targets):
    adj = adjacency(n, edges)
    seen = [False] * n
    seen[start] = True
    q = deque([start])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True
                q.append(y)
    return all(seen[t] for t in targets)


def bfs_path(n, edges, s, t):
    """Deterministic BFS shortest path (list of (u,v) directed edges) s -> t."""
    adj = adjacency(n, edges)
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


def make_edge_index(coords, edges):
    idx = {}
    for i, (u, v) in enumerate(edges):
        idx[(coords[u], coords[v])] = i
        idx[(coords[v], coords[u])] = i
    return idx


def all_pairs(term_nodes):
    out = []
    k = len(term_nodes)
    for i in range(k):
        for j in range(i + 1, k):
            out.append((term_nodes[i], term_nodes[j]))
    return out


# ---- current-injection electrical solve (Tero-style physarum network) ----
def solve_pair_np(np, n, edges, D, s, t):
    """Inject +1 unit of current at s, extract at t (grounded, V[t]=0). This
    current-source boundary condition (not a fixed-voltage one) is what keeps
    edge flux invariant under a uniform rescaling of D -- the property that
    lets the reinforcement dynamics below actually differentiate a sparse
    network instead of decaying everything uniformly toward zero."""
    L = np.zeros((n, n))
    for (u, v), d in zip(edges, D):
        L[u, u] += d
        L[v, v] += d
        L[u, v] -= d
        L[v, u] -= d
    free = [i for i in range(n) if i != t]
    idxmap = {node: i for i, node in enumerate(free)}
    Lred = L[np.ix_(free, free)] + np.eye(len(free)) * 1e-9
    Ivec = np.zeros(len(free))
    Ivec[idxmap[s]] = 1.0
    Vfree = np.linalg.solve(Lred, Ivec)
    V = np.zeros(n)
    for node, i in idxmap.items():
        V[node] = Vfree[i]
    V[t] = 0.0
    return V


def run_sim_np(np, n, edges, pairs, D0, mu, rounds=ROUNDS, alpha=ALPHA):
    """Deterministic physarum tube reinforcement. Each round: for every
    terminal pair, solve the current-injection network for the CURRENT
    conductances D, and accumulate |flux| across pairs onto each edge BEFORE
    applying the mu-exponent (this pre-exponent summation is what lets a
    junction edge shared by several pairs' flow get a super-additive boost
    when mu > 1 -- the "feedback exponent that merges flows"). Then relax D
    toward Qtot^mu and clip to [D_MIN, D_MAX]."""
    D = np.array(D0, dtype=float)
    m = len(edges)
    for _ in range(rounds):
        Qtot = np.zeros(m)
        for (s, t) in pairs:
            V = solve_pair_np(np, n, edges, D, s, t)
            Q = np.array([D[i] * abs(V[u] - V[v]) for i, (u, v) in enumerate(edges)])
            Qtot += Q
        target = np.power(np.clip(Qtot, 0.0, None), mu)
        D = (1.0 - alpha) * D + alpha * target
        D = np.clip(D, D_MIN, D_MAX)
    return D


def network_edges(edges, D, thresh=THRESH):
    return [(u, v) for i, (u, v) in enumerate(edges) if D[i] >= thresh]


def network_length(edges, D, thresh=THRESH):
    return sum(1 for i in range(len(edges)) if D[i] >= thresh)


def polar_terminals(R, C, k, radius, phase, jitter_fn):
    """k terminals placed on a circle of the given radius around grid center,
    angularly spaced 2*pi/k apart with a small deterministic per-index jitter."""
    cr, cc = R // 2, C // 2
    pts = []
    for i in range(k):
        theta = phase + 2.0 * math.pi * i / k
        dr, dc = jitter_fn(i)
        r = cr + round(radius * math.sin(theta)) + dr
        c = cc + round(radius * math.cos(theta)) + dc
        r = max(0, min(R - 1, r))
        c = max(0, min(C - 1, c))
        pts.append((r, c))
    return pts
