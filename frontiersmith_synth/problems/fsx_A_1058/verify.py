#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for 'Trunk Pricing'.

Validates feasibility strictly (capacities, conservation, global backbone
budget, finite integer tokens, exact token count), then scores
    score = min(1, B / (10*F))
where F is the participant's total cost and B is the checker's own baseline:
the cheapest way to satisfy every commodity using ONLY its non-trunk edges
(never touching the shared backbone at all).
"""
import sys
from collections import deque


def mcmf(n, edges, s, t, need):
    """edges: list of (u, v, cap, cost). Successive shortest augmenting paths
    (SPFA/Bellman-Ford, handles the negative reduced-cost residual arcs that
    appear as augmentation proceeds). Returns (achieved_flow, per_edge_flow)."""
    graph = [[] for _ in range(n)]  # arc = [to, cap, cost, rev_idx]
    arc_of = [None] * len(edges)
    for i, (u, v, cap, cost) in enumerate(edges):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        arc_of[i] = (u, len(graph[u]) - 1)

    flow = 0
    INF = float("inf")
    while flow < need:
        dist = [INF] * n
        dist[s] = 0.0
        inq = [False] * n
        pv = [None] * n
        dq = deque([s])
        inq[s] = True
        while dq:
            u = dq.popleft()
            inq[u] = False
            du = dist[u]
            for idx, arc in enumerate(graph[u]):
                v, cap, cost, _rev = arc
                if cap > 0 and du + cost < dist[v] - 1e-9:
                    dist[v] = du + cost
                    pv[v] = (u, idx)
                    if not inq[v]:
                        dq.append(v)
                        inq[v] = True
        if dist[t] == INF:
            break
        push = need - flow
        v = t
        while v != s:
            u, idx = pv[v]
            push = min(push, graph[u][idx][1])
            v = u
        v = t
        while v != s:
            u, idx = pv[v]
            graph[u][idx][1] -= push
            graph[v][graph[u][idx][3]][1] += push
            v = u
        flow += push

    flows = []
    for i in range(len(edges)):
        au, aidx = arc_of[i]
        cap_orig = edges[i][2]
        flows.append(cap_orig - graph[au][aidx][1])
    return flow, flows


def fail(reason):
    print(f"Infeasible: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(toks[p])
        p += 1
        return v

    K = nxt()
    commodities = []
    for _ in range(K):
        n = nxt(); m = nxt(); s = nxt(); t = nxt(); d = nxt()
        edges = []
        for _ in range(m):
            u = nxt(); v = nxt(); cap = nxt(); cost = nxt(); shared = nxt(); weight = nxt()
            edges.append((u, v, cap, cost, shared, weight))
        commodities.append({"n": n, "s": s, "t": t, "d": d, "edges": edges})
    C = nxt()
    return commodities, C


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    commodities, C = read_instance(in_path)
    total_edges = sum(len(c["edges"]) for c in commodities)

    raw = open(out_path).read().split()
    if len(raw) != total_edges:
        fail(f"expected {total_edges} integers, got {len(raw)}")

    flows_by_commodity = []
    pos = 0
    for c in commodities:
        m = len(c["edges"])
        vals = []
        for tok in raw[pos:pos + m]:
            # Reject anything that isn't a plain finite decimal integer literal
            # BEFORE calling int() -- int() happily parses arbitrarily long
            # digit strings (and math.isinf() on a huge int would itself raise
            # OverflowError), so bound the token length first. Every real flow
            # value here is bounded by a two-digit capacity, so 18 digits is
            # already far beyond anything a valid submission could need.
            # NOTE: str.isdigit() is too permissive -- it accepts non-ASCII
            # "digit" characters (e.g. superscript '²') that int() itself
            # then rejects, so check against the ASCII digit set explicitly.
            core = tok[1:] if tok[:1] in "+-" else tok
            if not core or any(ch not in "0123456789" for ch in core) or len(core) > 18:
                fail(f"non-integer/out-of-range token '{tok}'")
                return 0
            fv = int(tok)
            vals.append(fv)
        pos += m
        flows_by_commodity.append(vals)

    # 1) capacity + non-negativity
    for ci, c in enumerate(commodities):
        for ei, (u, v, cap, cost, shared, weight) in enumerate(c["edges"]):
            fv = flows_by_commodity[ci][ei]
            if fv < 0 or fv > cap:
                fail(f"commodity {ci} edge {ei}: flow {fv} out of [0,{cap}]")

    # 2) flow conservation + demand
    for ci, c in enumerate(commodities):
        n, s, t, d = c["n"], c["s"], c["t"], c["d"]
        net = [0] * n  # inflow - outflow
        for ei, (u, v, cap, cost, shared, weight) in enumerate(c["edges"]):
            fv = flows_by_commodity[ci][ei]
            net[v] += fv
            net[u] -= fv
        for node in range(n):
            if node == s:
                if net[node] != -d:
                    fail(f"commodity {ci}: source net flow {-net[node]} != demand {d}")
            elif node == t:
                if net[node] != d:
                    fail(f"commodity {ci}: sink net inflow {net[node]} != demand {d}")
            else:
                if net[node] != 0:
                    fail(f"commodity {ci}: node {node} conservation violated ({net[node]})")

    # 3) global coupling constraint
    total_backbone = 0
    for ci, c in enumerate(commodities):
        for ei, (u, v, cap, cost, shared, weight) in enumerate(c["edges"]):
            if shared:
                total_backbone += flows_by_commodity[ci][ei] * weight
    if total_backbone > C:
        fail(f"global backbone usage {total_backbone} exceeds budget {C}")

    # objective
    F = 0.0
    for ci, c in enumerate(commodities):
        for ei, (u, v, cap, cost, shared, weight) in enumerate(c["edges"]):
            F += flows_by_commodity[ci][ei] * cost

    # baseline B: best cost using ONLY non-trunk (shared=0) edges per commodity
    B = 0.0
    for c in commodities:
        edges_notrunk = [(u, v, (cap if shared == 0 else 0), cost)
                          for (u, v, cap, cost, shared, weight) in c["edges"]]
        achieved, flows = mcmf(c["n"], edges_notrunk, c["s"], c["t"], c["d"])
        if achieved < c["d"]:
            # defensive fallback (should never trigger given the generator's
            # guarantee of a trunk-free path with capacity >= d); avoid a
            # non-positive/zero baseline.
            worst_cost = max(cost for (_u, _v, _cap, cost) in edges_notrunk) or 1
            B += c["d"] * worst_cost
        else:
            B += sum(f * e[3] for f, e in zip(flows, edges_notrunk))

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
