import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0370 -- constrained-OR (Format B, isolated candidate)
# Theme: "pandemic contact net" -- allocate a limited stockpile of a critical
# supply (rapid test / antiviral courses) across a SUPPLY TREE of health
# districts to minimize expected (holding + cross-district escalation +
# unmet-case shortage) cost under a network fill-rate service constraint.
#
# Supply-tree semantics (guaranteed one-way downstream pooling):
#   node 0 = central depot (root, no local demand). Each other node i has a
#   parent parent[i]; a node's stock covers its OWN random demand first, and
#   any leftover can cover the UNMET demand escalated up from its subtree
#   (its descendants) -- risk pooling, but every escalated unit costs e.
#   Stock at a node can ONLY reach itself and its descendants (never siblings),
#   so WHERE in the tree you place stock matters. Demand still unmet at the
#   root is lost -> shortage penalty p. Stock left unused anywhere -> holding h.
#
# The candidate sees only the DISTRIBUTION (per-node mean/std) + tree + costs +
# budget + target; the actual seeded demand SCENARIOS are hidden. It must pick
# a robust integer allocation. Objective is normalized to the uniform-split
# baseline. Infeasible (over budget / below service target / ill-formed) -> 0.
# ==========================================================================


def _topo_children_first(parent):
    """Return node order with every child before its parent (deepest first)."""
    n = len(parent)
    depth = [0] * n
    for i in range(1, n):
        d = 0; j = i
        while parent[j] != -1:
            d += 1; j = parent[j]
        depth[i] = d
    return sorted(range(n), key=lambda i: -depth[i])


def _sim_cost(pub, scen, x):
    """Deterministic expected cost of allocation x over the hidden scenarios.
    Returns (avg_cost, network_fill_rate). Pure integer cascade."""
    parent = pub["parent"]; N = pub["N"]
    h, e, p = pub["h"], pub["e"], pub["p"]
    ordr = _topo_children_first(parent)
    tot_cost = 0.0; tot_dem = 0; tot_unmet = 0
    for d in scen:
        avail = [0] * (N + 1); incoming = [0] * (N + 1); lu = [0] * (N + 1)
        for i in range(1, N + 1):
            xi = x[i]; di = d[i]
            if xi >= di:
                avail[i] = xi - di
            else:
                lu[i] = di - xi
        avail[0] = x[0]
        leftover = 0; escalated = 0; root_unmet = 0
        for i in ordr:
            inc = incoming[i]; av = avail[i]
            cover = av if av < inc else inc      # cover descendant shortfall
            escalated += cover
            leftover += av - cover
            passup = (lu[i] if i > 0 else 0) + (inc - cover)
            pp = parent[i]
            if pp != -1:
                incoming[pp] += passup
            else:
                root_unmet = passup
        tot_cost += h * leftover + e * escalated + p * root_unmet
        tot_dem += sum(d); tot_unmet += root_unmet
    fill = 1.0 - tot_unmet / max(tot_dem, 1)
    return tot_cost / len(scen), fill


def _build_tree(rng, N):
    """Random rooted supply tree (node 0 = depot). Shallow-ish, branching."""
    parent = [-1] * (N + 1)
    for i in range(1, N + 1):
        parent[i] = rng.randint(0, i // 2)   # attach to depot or an early node
    return parent


def make_instances():
    # (N, Bfac, mean_lo, mean_hi):  budget factor < ~1.1 makes stockouts bite;
    # wide mean range makes the uniform split badly mismatched (headroom).
    specs = [
        (7,  1.12, 5,  32),
        (8,  1.10, 4,  36),
        (9,  1.08, 6,  34),
        (10, 1.12, 5,  30),
        (7,  1.06, 4,  38),   # tighter budget
        (9,  1.10, 8,  40),
        (11, 1.09, 5,  35),
        (8,  1.14, 6,  28),
        (10, 1.07, 4,  40),   # tight + wide -> hard / held-out flavor
        (12, 1.11, 5,  33),
    ]
    h, e, p = 1.0, 4.0, 25.0
    target = 0.80
    K = 200
    out = []
    for si, (N, Bfac, mlo, mhi) in enumerate(specs):
        rng = random.Random(4100 + 7 * si)
        parent = _build_tree(rng, N)
        means = [0.0]; stds = [0.0]
        for i in range(1, N + 1):
            m = rng.uniform(mlo, mhi)
            means.append(round(m, 4))
            stds.append(round(rng.uniform(0.25, 0.55) * m, 4))
        B = int(round(sum(means) * Bfac))
        srng = random.Random(90000 + 13 * si + 1)
        scen = []
        for _ in range(K):
            d = [0] * (N + 1)
            for i in range(1, N + 1):
                v = means[i] + stds[i] * srng.gauss(0.0, 1.0)
                d[i] = v if v > 0 else 0
                d[i] = int(round(d[i]))
            scen.append(d)
        pub = {
            "parent": parent, "means": means, "stds": stds,
            "N": N, "B": B, "h": h, "e": e, "p": p,
            "target": target, "K": K,
        }
        out.append({"public": pub, "hidden": {"scen": scen}})
    return out


def _uniform_alloc(pub):
    N = pub["N"]; B = pub["B"]
    base = B // (N + 1)
    x = [base] * (N + 1)
    x[0] += B - base * (N + 1)          # remainder parked at the depot
    return x


def baseline(inst):
    pub = inst["public"]
    x = _uniform_alloc(pub)
    c, _ = _sim_cost(pub, inst["hidden"]["scen"], x)
    return c


def score(inst, ans):
    pub = inst["public"]; N = pub["N"]; B = pub["B"]
    if not isinstance(ans, dict) or "stock" not in ans:
        return False, 0.0
    stock = ans["stock"]
    if not isinstance(stock, list) or len(stock) != N + 1:
        return False, 0.0
    x = []
    total = 0
    for v in stock:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, 0.0
        fv = float(v)
        if fv != fv or fv in (float("inf"), float("-inf")):
            return False, 0.0
        if abs(fv - round(fv)) > 1e-6:      # must be integer units
            return False, 0.0
        iv = int(round(fv))
        if iv < 0 or iv > B:
            return False, 0.0
        total += iv
        x.append(iv)
    if total > B:                            # budget constraint
        return False, 0.0
    c, fill = _sim_cost(pub, inst["hidden"]["scen"], x)
    if fill < pub["target"] - 1e-9:          # service-level constraint
        return False, 0.0
    if c != c or c <= 0.0:
        return False, 0.0
    return True, c


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
