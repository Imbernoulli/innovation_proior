#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for Patient Zero (format C).

Instance (<in>):
    line1: testId
    line2: N M
    next M lines: u v w        (contact edge, transmission prob p=w/10, w in 1..9)
    line: T                    (rounds the SI cascade ran before the snapshot)
    line: K                    (number of infected nodes observed)
    line: c_1 ... c_K          (the infected node ids, ascending)

Submission (<out>): a plausibility WEIGHT per infected candidate, in the SAME
order c_1..c_K:
    line1: K'
    line2: K' non-negative reals w_1..w_K'

FEASIBILITY: K' must equal K; every weight finite and >= 0; the weights must
not all be zero. Any violation -> Ratio 0.0.

OBJECTIVE (maximize): normalise the weights to a distribution p_i = w_i / sum(w),
then F = p_{s0}, the probability mass the submission places on the HIDDEN true
index case s0. s0 is never given to the solver; verify.py recovers it by
re-running gen.py's exact deterministic construction from testId (and sanity
checks the reconstructed infected set matches <in> byte-for-byte).

BASELINE: B = 1/K, the mass a uniform "no idea" guess would place on s0.
Score = min(CAP, 0.1 * F / B), so a uniform submission scores ~0.1 and a
submission that concentrates >=10x the uniform share on the true source caps
out (below 1.0, so headroom is preserved for a policy that beats our
reference).
"""
import sys, math, random

EASY_IDS = {1, 2, 3, 4}
K_LADDER = {1: 6, 2: 6, 3: 7, 4: 7, 5: 8, 6: 8, 7: 9, 8: 9, 9: 10, 10: 10}
SPOKE_LO, SPOKE_HI = 4, 7
BRIDGE_W = [6, 7, 8, 9]
TREE_W = [6, 7, 8, 9]
EXTRA_W = [5, 6, 7]
MAX_ATTEMPTS = 5000

CAP = 0.9
MAX_TOKEN = 1e15


def out_ratio(v, reason=""):
    if reason:
        sys.stdout.write("# %s\n" % reason)
    sys.stdout.write("Ratio: %.6f\n" % v)
    sys.exit(0)


# ---------------- IDENTICAL construction to gen.py (regenerates the hidden s0) ----------------

def build_graph(test_id, attempt):
    rng = random.Random(1182_0000 + test_id * 97711 + attempt * 131 + 17)
    K = K_LADDER[test_id]
    spoke_sizes = [rng.randint(SPOKE_LO, SPOKE_HI) for _ in range(K)]
    node_id = 1
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
        return {"N": N, "edges": edges, "T": T, "infected": infected, "s0": s0}
    raise RuntimeError("verify: could not reconstruct test %d" % test_id)


# ---------------------------------------------------------------------------

def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        lines = f.read().split("\n")
    try:
        test_id = int(lines[0].strip())
        N, M = map(int, lines[1].split())
    except Exception:
        out_ratio(0.0, "bad instance header")
    if N <= 0 or M < 0 or M > 2_000_000:
        out_ratio(0.0, "bad N/M")
    edges = {}
    ptr = 2
    for _ in range(M):
        try:
            u, v, w = map(int, lines[ptr].split())
        except Exception:
            out_ratio(0.0, "bad edge line")
        ptr += 1
        if not (0 <= u < N and 0 <= v < N) or u == v or not (1 <= w <= 9):
            out_ratio(0.0, "edge out of range")
        edges[(min(u, v), max(u, v))] = w
    try:
        T = int(lines[ptr].strip()); ptr += 1
        K = int(lines[ptr].strip()); ptr += 1
        cand = list(map(int, lines[ptr].split())); ptr += 1
    except Exception:
        out_ratio(0.0, "bad T/K/candidates")
    if K <= 1 or K != len(cand) or sorted(cand) != cand or len(set(cand)) != K:
        out_ratio(0.0, "bad candidate list")
    if any(not (0 <= c < N) for c in cand):
        out_ratio(0.0, "candidate id out of range")

    # -- reconstruct hidden ground truth, sanity check against the instance file --
    try:
        truth = construct_instance(test_id)
    except Exception:
        out_ratio(0.0, "cannot reconstruct ground truth")
    if truth["N"] != N or sorted(truth["edges"].keys()) != sorted(edges.keys()):
        out_ratio(0.0, "instance does not match reconstructed generator output")
    if sorted(truth["infected"]) != cand or truth["T"] != T:
        out_ratio(0.0, "infected set / T mismatch with generator")
    s0 = truth["s0"]
    if s0 not in cand:
        out_ratio(0.0, "internal error: hidden source not infected")

    # -- read submission --
    try:
        with open(outf) as f:
            odata = f.read().split("\n")
    except Exception:
        out_ratio(0.0, "no output")
    try:
        kp = int(odata[0].strip())
    except Exception:
        out_ratio(0.0, "bad K'")
    if kp != K:
        out_ratio(0.0, "K' != K")
    if len(odata) < 2:
        out_ratio(0.0, "missing weight line")
    toks = odata[1].split()
    if len(toks) != K:
        out_ratio(0.0, "wrong weight count")
    weights = []
    for t in toks:
        try:
            x = float(t)
        except Exception:
            out_ratio(0.0, "non-numeric weight")
        if not math.isfinite(x):
            out_ratio(0.0, "non-finite weight")
        if x < 0.0 or x > MAX_TOKEN:
            out_ratio(0.0, "weight out of range")
        weights.append(x)
    total = sum(weights)
    if not (total > 0.0) or not math.isfinite(total):
        out_ratio(0.0, "weights sum to zero (or overflow)")

    idx = cand.index(s0)
    p_s0 = weights[idx] / total

    B = 1.0 / K
    F = p_s0
    ratio = min(1.0, 0.1 * F / B)
    ratio = min(ratio, CAP)
    out_ratio(ratio)


if __name__ == "__main__":
    main()
