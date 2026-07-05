import sys, json, random, isorun

# =============================================================================
# Vineyard Irrigation -- Season-Weighted Moisture-Predicate Prefix-Cache Ordering
# (Format B, isolated)   family: offline-decision-policy   variant 5, "large"
#
# A vineyard's irrigation controller runs a fixed library of "irrigation decision
# checks". Each check evaluates a SET of moisture/canopy sensor predicates. Over a
# whole growing SEASON each distinct check is scheduled a fixed number of times
# (its season weight w_q) -- some checks run every day, some only a few times.
#
# The controller evaluates predicates through a PREFIX CACHE (a trie). All
# predicates are laid out in ONE GLOBAL evaluation order, chosen once. Each check
# then evaluates its own predicates in that global order; whenever a check's
# leading run of predicates matches the leading run of an EARLIER-seen check, that
# shared prefix is a cache HIT and its token cost is reimbursed for free.
#
# NEW vs a plain hit-token layout: the hit tokens of a check are counted
# w_q times (once per scheduled run in the season). So the best global order must
# line up the shared cores of the HIGH-FREQUENCY check families at the front --
# a layout tuned only to raw occurrence counts (ignoring season weight) leaves
# reuse on the table. The candidate returns only a permutation; the evaluator
# replays the weighted stream through the trie itself to compute the score.
# =============================================================================


def _build_instance(seed, M, T, n_groups, gsize_lo, gsize_hi, nspr, p_drop):
    # Predicates 0..nspr-1 are cheap, RARE "sprinkle" predicates (per-check quirks);
    # predicates nspr..M-1 are expensive, FREQUENT "core" predicates shared by check
    # families. Sprinkles carry LOW indices, so the identity layout sorts them to the
    # FRONT of every check -- shattering the shared prefix. A layout that pushes the
    # frequent, heavy core predicates (weighted by season frequency) first recovers
    # long shared prefixes.
    rng = random.Random(seed)
    weights = [0] * M
    for a in range(nspr):
        weights[a] = rng.randint(1, 2)
    for a in range(nspr, M):
        weights[a] = rng.randint(3, 6)
    core = list(range(nspr, M))
    # each check family = a random subset of core predicates + a season frequency
    season_pool = [1, 1, 2, 3, 5, 8]
    groups = []
    gweights = []
    for _ in range(n_groups):
        size = rng.randint(gsize_lo, min(gsize_hi, len(core)))
        groups.append(sorted(rng.sample(core, size)))
        gweights.append(rng.choice(season_pool))
    queries = []
    qweights = []
    for _ in range(T):
        gi = rng.randrange(n_groups)
        g = groups[gi]
        s = set(g)
        if len(s) > 2 and rng.random() < p_drop:
            s.discard(g[rng.randrange(len(g))])
        for _ in range(rng.randint(1, 2)):
            s.add(rng.randrange(nspr))  # 1-2 sprinkle predicates
        queries.append(sorted(s))
        # per-run season weight: family base frequency jittered a little
        qw = gweights[gi] + rng.randint(0, 2)
        qweights.append(qw)
    return {"public": {"M": M, "weights": weights,
                       "queries": queries, "qweights": qweights},
            "hidden": {}}


def make_instances():
    specs = [
        # seed,  M,   T, ngrp, glo, ghi, nspr, p_drop
        (9101, 24,  70, 5, 4, 7, 12, 0.25),
        (9102, 26,  85, 6, 4, 7, 13, 0.25),
        (9103, 28,  95, 6, 5, 8, 14, 0.30),
        (9104, 25,  80, 5, 5, 8, 12, 0.20),
        (9105, 30, 110, 7, 4, 8, 15, 0.30),  # noisier
        (9106, 30, 120, 7, 5, 9, 14, 0.30),  # harder
        (9107, 24,  72, 5, 4, 7, 12, 0.25),
        (9108, 32, 140, 8, 5, 9, 16, 0.35),  # hardest / held-out
        (9109, 27,  90, 6, 4, 7, 13, 0.20),
        (9110, 28, 100, 6, 5, 8, 13, 0.28),
        (9111, 26,  88, 6, 4, 8, 12, 0.30),
    ]
    return [_build_instance(*sp) for sp in specs]


def _objective(M, weights, queries, qweights, order):
    # rank[a] = position of predicate a in the global order
    rank = [0] * M
    for pos, a in enumerate(order):
        rank[a] = pos
    trie = {}
    hit = 0
    for q, qw in zip(queries, qweights):
        seq = sorted(q, key=lambda a: rank[a])
        node = trie
        matched = 0
        broke = False
        for a in seq:
            if (not broke) and (a in node):
                matched += weights[a]
                node = node[a]
            else:
                broke = True
                nxt = node.get(a)
                if nxt is None:
                    nxt = {}
                    node[a] = nxt
                node = nxt
        hit += qw * matched
    return hit


def baseline(inst):
    p = inst["public"]
    M = p["M"]
    obj = _objective(M, p["weights"], p["queries"], p["qweights"], list(range(M)))
    return max(obj, 1)  # >=1 by construction (repeated identical checks share)


def score(inst, ans):
    p = inst["public"]
    M = p["M"]
    if not isinstance(ans, dict) or "order" not in ans:
        return False, 0
    order = ans["order"]
    if not isinstance(order, list) or len(order) != M:
        return False, 0
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in order):
        return False, 0
    if sorted(order) != list(range(M)):
        return False, 0
    obj = _objective(M, p["weights"], p["queries"], p["qweights"], order)
    if not (obj == obj):  # nan guard
        return False, 0
    return True, obj


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
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
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
