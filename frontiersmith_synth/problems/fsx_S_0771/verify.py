#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the truss-build-order problem.

Feasibility (checked at EVERY prefix of the submitted build order):
  every "erected" non-anchor node (touches >=1 currently built member) must be
  able to route its 1 unit of self-weight to SOME anchor through the currently
  built members only, respecting each member's capacity -- verified via exact
  integer max-flow (source -> each erected node cap 1, members undirected cap,
  anchors -> sink cap INF). Any prefix where max-flow < erected-count is a
  structural collapse -> whole submission scores 0.

Objective (final structure only): max-flow from a super-source touching every
deck node (cap INF) to a super-sink touching every anchor (cap INF), through
the FINAL member set (their full capacities). This is the bridge's rateable
load capacity.

Baseline B: for each deck node, the checker's own widest (max-bottleneck) path
to any anchor over the FULL candidate graph; union of those paths' members.
"""
import sys
from collections import deque

INF = 10 ** 9


def max_flow(n, source, sink, cap):
    """cap: dict[(u,v)] -> residual capacity (mutated in place). Returns flow value."""
    flow = 0
    while True:
        parent = {source: None}
        dq = deque([source])
        found = False
        while dq:
            u = dq.popleft()
            if u == sink:
                found = True
                break
            for (a, b), c in list(cap.items()):
                if a == u and c > 0 and b not in parent:
                    parent[b] = u
                    dq.append(b)
        if not found:
            break
        path = []
        v = sink
        while parent[v] is not None:
            u = parent[v]
            path.append((u, v))
            v = u
        bottleneck = min(cap[(u, v)] for (u, v) in path)
        for (u, v) in path:
            cap[(u, v)] -= bottleneck
            cap[(v, u)] = cap.get((v, u), 0) + bottleneck
        flow += bottleneck
    return flow


def undirected_caps(edge_list):
    cap = {}
    for (u, v, c) in edge_list:
        cap[(u, v)] = cap.get((u, v), 0) + c
        cap[(v, u)] = cap.get((v, u), 0) + c
    return cap


def self_weight_feasible(built_edges, anchors):
    """built_edges: list of (u,v,cap) currently built. Returns True if every
    erected non-anchor node can route 1 unit to an anchor."""
    erected = set()
    for (u, v, c) in built_edges:
        if u not in anchors:
            erected.add(u)
        if v not in anchors:
            erected.add(v)
    if not erected:
        return True
    SRC, SNK = "__S__", "__T__"
    cap = undirected_caps(built_edges)
    for node in erected:
        cap[(SRC, node)] = cap.get((SRC, node), 0) + 1
    for a in anchors:
        cap[(a, SNK)] = cap.get((a, SNK), 0) + INF
    f = max_flow(None, SRC, SNK, cap)
    return f >= len(erected)


def flow_value(edge_list, sources, sinks):
    SRC, SNK = "__S__", "__T__"
    cap = undirected_caps(edge_list)
    for s in sources:
        cap[(SRC, s)] = cap.get((SRC, s), 0) + INF
    for t in sinks:
        cap[(t, SNK)] = cap.get((t, SNK), 0) + INF
    return max_flow(None, SRC, SNK, cap)


def straight_column_baseline(edges, W, H, deck):
    """The dumbest safe reference: for each deck node, use ONLY the vertical
    dx=0 riser chain straight down to the anchor directly beneath it. Deck
    x-positions are distinct, so these chains never share a member -- always
    self-weight feasible in isolation (each chain's demand tops out at H,
    which the generator keeps within every riser's capacity)."""
    lookup = {}
    for idx, (u, v, c) in enumerate(edges):
        lookup[(u, v)] = idx
        lookup[(v, u)] = idx

    def node_id(x, y):
        return y * (W + 1) + x

    used_idx = set()
    for d in deck:
        x = d % (W + 1)
        y = d // (W + 1)
        while y > 0:
            u, v = node_id(x, y - 1), node_id(x, y)
            key = (u, v) if (u, v) in lookup else (v, u)
            used_idx.add(lookup[key])
            y -= 1
    return [edges[i] for i in used_idx]


def fail(reason):
    print("collapse:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf_path, out_path = sys.argv[1], sys.argv[2]

    with open(inf_path) as f:
        toks = f.read().split()
    p = 0
    W, H, M, D = (int(toks[p + i]) for i in range(4)); p += 4
    edges = []
    for _ in range(M):
        u, v, c = int(toks[p]), int(toks[p + 1]), int(toks[p + 2]); p += 3
        edges.append((u, v, c))
    deck = [int(toks[p + i]) for i in range(D)]; p += D
    anchors = set(range(0, W + 1))

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except Exception:
        fail("cannot read output")

    if not out_toks:
        fail("empty output")

    try:
        K = int(out_toks[0])
    except ValueError:
        fail("K not an integer")

    if K < 0 or K > M:
        fail(f"K={K} out of range [0,{M}]")
    if len(out_toks) != 1 + K:
        fail(f"expected {1+K} tokens, got {len(out_toks)}")

    idxs = []
    seen = set()
    for i in range(K):
        try:
            v = int(out_toks[1 + i])
        except ValueError:
            fail(f"token {1+i} not an integer")
        if v < 0 or v >= M:
            fail(f"member index {v} out of range [0,{M-1}]")
        if v in seen:
            fail(f"member index {v} used twice")
        seen.add(v)
        idxs.append(v)

    # simulate the build, checking self-weight feasibility at every prefix
    built = []
    for step, idx in enumerate(idxs):
        built.append(edges[idx])
        if not self_weight_feasible(built, anchors):
            fail(f"self-weight collapse at step {step} adding member {idx}")

    final_edges = built
    F = flow_value(final_edges, deck, anchors)

    baseline_edges = straight_column_baseline(edges, W, H, deck)
    B = flow_value(baseline_edges, deck, anchors)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"F={F} B={B} K={K}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
