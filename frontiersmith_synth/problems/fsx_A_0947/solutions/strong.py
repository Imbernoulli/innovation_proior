# TIER: strong
"""Exact-min-cut insight.

For every zone (each protected basin, and the forbidden zone) compute the true
minimum edge cut, via max-flow, that separates the source (elevation-0 water)
from that zone in the graph of cells ever reachable by the final water level.
Max-flow/min-cut automatically ignores any cell that is close to the zone by raw
distance but not actually on a path to it (a dead-end notch carries no flow, so
it never appears in the cut) -- this is exactly what the reactive, distance-only
heuristic cannot tell apart from a real corridor.

Since the total wall budget may not cover every zone, try every subset of zones
(there are only a handful) and, for each, check whether its combined cut edges
can be scheduled -- each edge built at or before the stage at which both its
endpoints first become elevation-eligible -- within the staged, cumulative
budget. Keep the feasible subset of maximum total value (protected zones weigh
1, the forbidden zone weighs alpha). This is a decomposition + budgeted
selection, not "greedy plus more iterations"."""
import sys
from collections import deque

INF = 10 ** 9


def min_cut_edges(R, C, elev, LK, sources, targets):
    elig = [[elev[r][c] <= LK for c in range(C)] for r in range(R)]
    cap = {}
    adj = {}

    def add_edge(u, v, c):
        cap[(u, v)] = cap.get((u, v), 0) + c
        cap.setdefault((v, u), 0)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    for r in range(R):
        for c in range(C):
            if not elig[r][c]:
                continue
            u = (r, c)
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and elig[nr][nc]:
                    v = (nr, nc)
                    add_edge(u, v, 1)
                    add_edge(v, u, 1)

    S, T = "S", "T"
    for (r, c) in sources:
        if elig[r][c]:
            add_edge(S, (r, c), INF)
    for (r, c) in targets:
        if elig[r][c]:
            add_edge((r, c), T, INF)

    if S not in adj or T not in adj:
        return []

    maxflow = 0
    while True:
        parent = {S: None}
        dq = deque([S])
        found = False
        while dq:
            u = dq.popleft()
            if u == T:
                found = True
                break
            for v in adj.get(u, ()):
                if v not in parent and cap.get((u, v), 0) > 0:
                    parent[v] = u
                    dq.append(v)
        if not found:
            break
        v = T
        bottleneck = INF
        while parent[v] is not None:
            u = parent[v]
            bottleneck = min(bottleneck, cap[(u, v)])
            v = u
        v = T
        while parent[v] is not None:
            u = parent[v]
            cap[(u, v)] -= bottleneck
            cap[(v, u)] += bottleneck
            v = u
        maxflow += bottleneck

    visited = {S}
    dq = deque([S])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, ()):
            if v not in visited and cap.get((u, v), 0) > 0:
                visited.add(v)
                dq.append(v)

    cut = set()
    for u in visited:
        if u in (S, T):
            continue
        for v in adj.get(u, ()):
            if v not in visited and v != T and cap.get((u, v), 0) == 0:
                cut.add(frozenset((u, v)))
    return [tuple(sorted(e)) for e in cut]


def try_schedule(edge_deadline_list, CumW, K):
    items = sorted(edge_deadline_list, key=lambda x: x[1])
    assigned = []
    count_so_far = 0
    for (edge, d) in items:
        need = count_so_far + 1
        chosen = None
        for s in range(1, d + 1):
            if CumW[s - 1] >= need:
                chosen = s
                break
        if chosen is None:
            return None
        assigned.append((edge, chosen))
        count_so_far += 1
    return assigned


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it)); K = int(next(it))
    elev = [[int(next(it)) for _ in range(C)] for _ in range(R)]
    NB = int(next(it))
    p_groups = []
    for _ in range(NB):
        sz = int(next(it))
        p_groups.append([(int(next(it)), int(next(it))) for _ in range(sz)])
    Q = int(next(it))
    f_cells = [(int(next(it)), int(next(it))) for _ in range(Q)]
    levels = [int(next(it)) for _ in range(K)]
    W = [int(next(it)) for _ in range(K)]
    alpha = float(next(it))

    CumW = []
    s = 0
    for k in range(K):
        s += W[k]
        CumW.append(s)

    sources = [(r, c) for r in range(R) for c in range(C) if elev[r][c] <= 0]
    LK = levels[-1]

    zones = []  # (value, edges)
    for grp in p_groups:
        edges = min_cut_edges(R, C, elev, LK, sources, grp)
        zones.append((1.0, edges))
    f_edges = min_cut_edges(R, C, elev, LK, sources, f_cells)
    zones.append((alpha, f_edges))

    n = len(zones)

    def edge_deadline(e):
        (r1, c1), (r2, c2) = e
        need = max(elev[r1][c1], elev[r2][c2])
        for k in range(K):
            if levels[k] >= need:
                return k + 1
        return K  # extremely conservative fallback; shouldn't happen for a real cut edge

    best_value = -1.0
    best_plan = []
    best_edge_count = None
    for mask in range(1, 1 << n):
        val = 0.0
        edge_list = []
        for i in range(n):
            if mask & (1 << i):
                v, edges = zones[i]
                val += v
                for e in edges:
                    edge_list.append((e, edge_deadline(e)))
        plan = try_schedule(edge_list, CumW, K)
        if plan is None:
            continue
        if val > best_value + 1e-9 or (abs(val - best_value) <= 1e-9 and
                                        (best_edge_count is None or len(plan) < best_edge_count)):
            best_value = val
            best_plan = plan
            best_edge_count = len(plan)

    out = [str(len(best_plan))]
    for (((r1, c1), (r2, c2)), st) in best_plan:
        out.append(f"{st} {r1} {c1} {r2} {c2}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
