import sys, json, random, isorun

# ==========================================================================
# fsx_C_0194 -- causal-discovery (Format B, isolated candidate)
# Theme: "deep-sea cable network".
#
# A deep-sea fibre-optic cable network is a set of n signal stations. A latent
# fault/attenuation disturbance that hits one station propagates DOWNSTREAM to
# the stations it directly feeds, and each station also picks up its own local
# noise.  Physically this is a linear-Gaussian structural causal model on a
# DAG: for each station j,
#     X_j = sum_{i -> j} w_{ij} X_i + eps_j ,   eps_j ~ N(0, sigma^2)
# with EQUAL disturbance variance sigma^2 at every station (identical optical
# repeaters), which makes the true propagation DAG identifiable from purely
# observational readings.
#
# You are handed a batch of observational signal readings (many independent
# snapshots of all n stations) and must RECONSTRUCT the directed propagation
# graph (who feeds whom).  You never see the true graph; the grader keeps it
# hidden and scores your answer by Structural Hamming Distance (SHD): the number
# of edge insertions, deletions and reversals needed to turn your graph into the
# true one.  Objective: MINIMIZE mean SHD over a battery of ground-truth
# networks.  An empty guess pays SHD = (#true edges); a cyclic / malformed guess
# scores 0.  The score is normalized so the empty-graph guess ~= 0.1.
# ==========================================================================


def _make_scm(n, p, m, seed):
    """Sample a random equal-variance linear-Gaussian DAG and m observations.
    Returns (edges, data) where edges is a list of (i,j) meaning i->j, and data
    is m rows of n floats."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    pos = {node: k for k, node in enumerate(order)}
    weights = {}
    edges = []
    for a in range(n):
        for b in range(n):
            if pos[a] < pos[b] and rng.random() < p:
                w = rng.uniform(0.4, 1.0) * (1.0 if rng.random() < 0.5 else -1.0)
                weights[(a, b)] = w
                edges.append((a, b))
    # guarantee at least one edge (so baseline SHD > 0)
    if not edges:
        a, b = order[0], order[1]
        weights[(a, b)] = 0.9
        edges.append((a, b))
    sigma = 1.0
    # incoming-parent lists in topological order for fast sampling
    parents = {b: [(a, w) for (a, bb), w in weights.items() if bb == b] for b in range(n)}
    data = []
    for _ in range(m):
        vals = {}
        for node in order:                       # parents precede children
            v = rng.gauss(0.0, sigma)
            for (a, w) in parents[node]:
                v += w * vals[a]
            vals[node] = v
        data.append([round(vals[k], 6) for k in range(n)])
    return edges, data


def make_instances():
    # (n, edge-prob p, #samples m) -- later, denser, sample-starved rows are
    # deliberately harder so a strong recovery method keeps headroom (< 1.0).
    specs = [
        (10, 0.28, 200),
        (10, 0.34, 140),
        (12, 0.28, 220),
        (12, 0.34, 150),
        (12, 0.40, 110),
        (13, 0.30, 160),
        (13, 0.36, 120),
        (14, 0.30, 180),
        (14, 0.35, 130),
        (14, 0.40, 100),
        (15, 0.34, 120),
        (15, 0.40, 90),
    ]
    out = []
    for si, (n, p, m) in enumerate(specs):
        edges, data = _make_scm(n, p, m, 40000 + 137 * si)
        pub = {"n": n, "samples": data}
        hid = {"edges": [list(e) for e in edges]}
        out.append({"public": pub, "hidden": hid})
    return out


def baseline(inst):
    # trivial construction = predict NO edges -> SHD equals the number of true
    # edges (every true edge is a missed edge).
    return float(len(inst["hidden"]["edges"]))


def _is_acyclic(n, edges):
    adj = {i: [] for i in range(n)}
    indeg = [0] * n
    for (i, j) in edges:
        adj[i].append(j)
        indeg[j] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return seen == n


def _shd(n, pred, true):
    ps = set(pred)
    ts = set(true)
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            p_state = 1 if (i, j) in ps else (2 if (j, i) in ps else 0)
            t_state = 1 if (i, j) in ts else (2 if (j, i) in ts else 0)
            if p_state != t_state:
                c += 1
    return c


def score(inst, ans):
    n = inst["public"]["n"]
    true = [tuple(e) for e in inst["hidden"]["edges"]]
    if not isinstance(ans, dict) or "edges" not in ans:
        return False, 0.0
    raw = ans["edges"]
    if not isinstance(raw, list):
        return False, 0.0
    pred = []
    seen = set()
    for e in raw:
        if (not isinstance(e, (list, tuple))) or len(e) != 2:
            return False, 0.0
        i, j = e
        if isinstance(i, bool) or isinstance(j, bool):
            return False, 0.0
        if (not isinstance(i, int)) or (not isinstance(j, int)):
            return False, 0.0
        if i < 0 or j < 0 or i >= n or j >= n or i == j:
            return False, 0.0
        if (i, j) in seen or (j, i) in seen:      # duplicate / both directions
            return False, 0.0
        seen.add((i, j))
        pred.append((i, j))
    if not _is_acyclic(n, pred):                  # a propagation graph must be a DAG
        return False, 0.0
    return True, float(_shd(n, pred, true))


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
