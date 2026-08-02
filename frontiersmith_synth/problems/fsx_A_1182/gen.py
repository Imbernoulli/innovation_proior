#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE instance of Patient Zero (format C).

A hidden "index case" s0 starts a discrete-time SI (susceptible-infected, no
recovery) cascade on a contact network: every round, every currently-infected
node independently infects each still-susceptible neighbour v across edge
(u,v) with probability p_uv = w_uv/10 (w_uv is an integer 1..9 printed on the
edge). After T rounds we STOP and print only the resulting infected SET
(snapshot-only-observation) -- never s0, never per-node infection times.

The network is a hub-and-spoke contact graph: one hub node bridges K disjoint
small communities ("spokes"). testId 1..4 place s0 AT the hub (an easy case:
the true origin genuinely is the best-connected node). testId 5..10 place s0
deep inside one *peripheral* spoke, adjacent to nothing but its own small
community (a TRAP case: raw degree/centrality crowns the hub, which merely
happens to be reachable from everywhere, while the true origin is a low-degree
node whose entire neighbourhood -- and nothing beyond -- got converted).

Everything is seeded by testId only (+ a bounded, deterministic retry counter)
-> fully reproducible. s0 itself is NEVER printed; verify.py reconstructs it
by re-running this exact same construction.
"""
import sys, random

EASY_IDS = {1, 2, 3, 4}
K_LADDER = {1: 6, 2: 6, 3: 7, 4: 7, 5: 8, 6: 8, 7: 9, 8: 9, 9: 10, 10: 10}
SPOKE_LO, SPOKE_HI = 4, 7
BRIDGE_W = [6, 7, 8, 9]
TREE_W = [6, 7, 8, 9]
EXTRA_W = [5, 6, 7]
MAX_ATTEMPTS = 5000


def build_graph(test_id, attempt):
    rng = random.Random(1182_0000 + test_id * 97711 + attempt * 131 + 17)
    K = K_LADDER[test_id]
    spoke_sizes = [rng.randint(SPOKE_LO, SPOKE_HI) for _ in range(K)]
    node_id = 1  # node 0 = hub
    spokes = []
    for sz in spoke_sizes:
        nodes = list(range(node_id, node_id + sz))
        node_id += sz
        spokes.append(nodes)
    N = node_id
    edges = {}

    def add_edge(u, v, w):
        key = (min(u, v), max(u, v))
        if key not in edges:
            edges[key] = w

    bridges = []
    for nodes in spokes:
        order = nodes[:]
        rng.shuffle(order)
        for i in range(1, len(order)):
            j = rng.randint(0, i - 1)
            add_edge(order[i], order[j], rng.choice(TREE_W))
        deg = {u: 0 for u in nodes}
        for (a, b) in edges:
            if a in deg: deg[a] += 1
            if b in deg: deg[b] += 1
        extra = rng.randint(0, max(0, len(nodes) - 3))
        for _ in range(extra):
            a, b = rng.sample(nodes, 2)
            if deg.get(a, 0) < 4 and deg.get(b, 0) < 4:
                key = (min(a, b), max(a, b))
                if key not in edges:
                    add_edge(a, b, rng.choice(EXTRA_W))
                    deg[a] += 1; deg[b] += 1
        bridge = rng.choice(nodes)
        bridges.append(bridge)
        add_edge(0, bridge, rng.choice(BRIDGE_W))

    return N, edges, spokes, bridges, rng


def spoke_diam(nodes, edges):
    nodeset = set(nodes)
    adj = {u: [] for u in nodes}
    for (a, b) in edges:
        if a in nodeset and b in nodeset:
            adj[a].append(b); adj[b].append(a)
    best = 0
    for s in nodes:
        dist = {s: 0}
        frontier = [s]
        while frontier:
            nf = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nf.append(v)
            frontier = nf
        best = max(best, max(dist.values()))
    return best


def simulate(N, edges, s0, T, rng):
    adj = {i: [] for i in range(N)}
    for (a, b), w in edges.items():
        p = w / 10.0
        adj[a].append((b, p))
        adj[b].append((a, p))
    infected = {s0}
    for _ in range(T):
        newly = set()
        for u in infected:
            for (v, p) in adj[u]:
                if v not in infected and rng.random() < p:
                    newly.add(v)
        infected |= newly
    return infected


def degree(N, edges):
    deg = {i: 0 for i in range(N)}
    for (a, b) in edges:
        deg[a] += 1; deg[b] += 1
    return deg


def construct_instance(test_id):
    """Deterministic construction shared verbatim by gen.py and verify.py.
    Returns dict with N, edges(dict (u,v)->w), T, infected(set), s0(hidden)."""
    mode = "easy" if test_id in EASY_IDS else "trap"
    for attempt in range(MAX_ATTEMPTS):
        N, edges, spokes, bridges, rng = build_graph(test_id, attempt)
        home = spokes[0]
        if mode == "easy":
            s0 = 0
            T = 3
        else:
            non_bridge = [u for u in home if u != bridges[0]]
            if not non_bridge:
                continue
            s0 = rng.choice(non_bridge)
            T = spoke_diam(home, edges) + 1

        simrng = random.Random(999_000 + test_id * 733 + attempt * 17 + 3)
        infected = simulate(N, edges, s0, T, simrng)
        if len(infected) < 8 or len(infected) > 60:
            continue
        deg = degree(N, edges)
        top = max(infected, key=lambda u: (deg[u], -u))
        if mode == "easy":
            if top != s0:
                continue
        else:
            if top == s0 or deg[top] < deg[s0] + 3:
                continue
        return {"N": N, "edges": edges, "T": T, "infected": infected, "s0": s0,
                "mode": mode, "K": K_LADDER[test_id]}
    raise RuntimeError("gen: no valid instance for test %d" % test_id)


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if test_id not in K_LADDER:
        test_id = ((test_id - 1) % 10) + 1
    inst = construct_instance(test_id)
    N = inst["N"]; edges = inst["edges"]; T = inst["T"]; infected = sorted(inst["infected"])

    out = sys.stdout
    out.write("%d\n" % test_id)
    out.write("%d %d\n" % (N, len(edges)))
    for (u, v), w in edges.items():
        out.write("%d %d %d\n" % (u, v, w))
    out.write("%d\n" % T)
    out.write("%d\n" % len(infected))
    out.write(" ".join(map(str, infected)))
    out.write("\n")


if __name__ == "__main__":
    main()
