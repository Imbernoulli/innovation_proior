import sys
from collections import deque

# Format D checker (op-count) -- Network-on-Chip topology design.
#   1) Parse instance: N, L_max, CAP, STALL_COST, traffic matrix T (<in>).
#   2) Parse participant topology: M undirected links (<out>).
#   3) Feasibility gate (validated strictly, any violation -> Ratio: 0.0):
#        - well-formed integer tokens only (rejects nan/inf/garbage)
#        - 0 <= M <= N*(N-1)/2, each edge in range, no self-loop, no duplicate
#        - sum of ring-distance link costs <= L_max
#        - graph spans (connects) all N nodes
#   4) Route every traffic pair by DETERMINISTIC canonical BFS (neighbors
#      visited in ascending id order -> a fixed shortest-path tree per source,
#      independent of the order links were listed in the output).
#   5) Op-count F = total traffic-weighted hop traversals + congestion-stall
#      penalty (edges whose accumulated load exceeds CAP pay STALL_COST per
#      unit over CAP). This is the FLOPs-style "operation count" analog: F
#      counts the link *operations* the simulated traffic performs.
#   6) Baseline B = the same op-count on the checker's own minimal ring
#      topology (0-1-2-...-N-1-0). Minimizing: Ratio = min(1, B/F) [0,1].

MAXM_FACTOR = 1  # M capped at N*(N-1)//2 (simple graph)

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

def read_ints(path):
    try:
        return [int(t) for t in open(path).read().split()]
    except Exception:
        return None

def ring_cost(i, j, N):
    d = abs(i - j)
    return min(d, N - d)

def build_adj(N, edges):
    adj = [[] for _ in range(N)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    for a in adj:
        a.sort()
    return adj

def compute_ops(N, adj, T, CAP, STALL_COST):
    """Deterministic canonical-BFS routing + op count for a fixed graph."""
    total_hops = 0
    load = {}  # frozenset-free key: (min(u,v), max(u,v)) -> accumulated traffic
    for src in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:          # adj[u] sorted ascending -> deterministic
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        row = T[src]
        for dst in range(N):
            w = row[dst]
            if w <= 0 or dst == src:
                continue
            d = dist[dst]
            # d must be finite: caller guarantees connectivity before calling
            total_hops += w * d
            # walk canonical path dst -> src, accumulate edge load
            cur = dst
            while cur != src:
                p = parent[cur]
                key = (p, cur) if p < cur else (cur, p)
                load[key] = load.get(key, 0) + w
                cur = p
    penalty = 0
    for key, ld in load.items():
        if ld > CAP:
            penalty += (ld - CAP) * STALL_COST
    return total_hops + penalty

def main():
    in_tok = read_ints(sys.argv[1])
    if in_tok is None or len(in_tok) < 4:
        fail("bad input file")
    it = iter(in_tok)
    N = next(it); L_max = next(it); CAP = next(it); STALL_COST = next(it)
    if not (2 <= N <= 200):
        fail("bad N")
    T = [[0] * N for _ in range(N)]
    try:
        for i in range(N):
            for j in range(N):
                T[i][j] = next(it)
    except StopIteration:
        fail("truncated traffic matrix")

    # ---- parse participant output ----
    raw = None
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    if not raw:
        fail("empty output")
    try:
        toks = [int(t) for t in raw]
    except Exception:
        fail("non-integer / non-finite token (nan/inf rejected)")

    if not toks:
        fail("empty token list")
    M = toks[0]
    maxM = N * (N - 1) // 2
    if M < 0 or M > maxM:
        fail("M out of range")
    need = 1 + 2 * M
    if len(toks) != need:
        fail("wrong token count (got %d, need %d)" % (len(toks), need))

    edges = []
    seen = set()
    total_cost = 0
    for k in range(M):
        u = toks[1 + 2 * k]; v = toks[2 + 2 * k]
        if not (0 <= u < N and 0 <= v < N) or u == v:
            fail("bad edge endpoint")
        key = (u, v) if u < v else (v, u)
        if key in seen:
            fail("duplicate edge")
        seen.add(key)
        edges.append(key)
        total_cost += ring_cost(u, v, N)

    if total_cost > L_max:
        fail("link budget exceeded (%d > %d)" % (total_cost, L_max))

    adj = build_adj(N, edges)
    # connectivity check (BFS from node 0)
    seen_nodes = [False] * N
    seen_nodes[0] = True
    q = deque([0])
    cnt = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen_nodes[v]:
                seen_nodes[v] = True
                cnt += 1
                q.append(v)
    if cnt != N:
        fail("topology is disconnected")

    F = compute_ops(N, adj, T, CAP, STALL_COST)
    if F <= 0:
        fail("degenerate zero traffic")

    ring_edges = [(i, (i + 1) % N) for i in range(N)]
    ring_edges = [(min(a, b), max(a, b)) for (a, b) in ring_edges]
    ring_adj = build_adj(N, ring_edges)
    B = compute_ops(N, ring_adj, T, CAP, STALL_COST)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("N=%d M=%d cost=%d B=%d F=%d Ratio: %.6f" % (N, M, total_cost, B, F, ratio))

if __name__ == "__main__":
    main()
