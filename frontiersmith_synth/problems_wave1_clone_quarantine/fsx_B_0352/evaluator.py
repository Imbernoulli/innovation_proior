import sys, json, random, isorun

# =============================================================================
# Greenhouse Zones -- Sensor-Predicate Prefix-Cache Ordering (Format B, isolated)
#
# A greenhouse controller processes a fixed stream of "irrigation control checks".
# Each check evaluates a SET of sensor predicates (attributes). The controller
# caches evaluation results as a PREFIX TREE (trie): when a new check's predicate
# sequence shares a leading prefix with an already-seen check, that prefix is a
# cache HIT (its token cost is saved). Predicates are laid out in ONE GLOBAL order
# chosen once; each check lists its own predicates filtered into that global order.
#
# The candidate chooses the global predicate ordering (a permutation). The
# evaluator itself replays the fixed stream through the trie and measures total
# hit tokens -- the candidate only returns an ordering, never a score.
# =============================================================================


def _build_instance(seed, M, T, n_groups, gsize_lo, gsize_hi, nspr, p_drop):
    # Attributes 0..nspr-1 are cheap, RARE "sprinkle" predicates (per-check quirks);
    # attributes nspr..M-1 are expensive, FREQUENT "core" predicates shared by check
    # families. Sprinkles carry LOW indices, so the identity layout sorts them to the
    # FRONT of every check -- shattering the shared prefix. A layout that pushes the
    # frequent, heavy core predicates first recovers long shared prefixes.
    rng = random.Random(seed)
    weights = [0] * M
    for a in range(nspr):
        weights[a] = rng.randint(1, 2)
    for a in range(nspr, M):
        weights[a] = rng.randint(3, 6)
    core = list(range(nspr, M))
    groups = []
    for _ in range(n_groups):
        size = rng.randint(gsize_lo, min(gsize_hi, len(core)))
        groups.append(sorted(rng.sample(core, size)))
    queries = []
    for _ in range(T):
        g = groups[rng.randrange(n_groups)]
        s = set(g)
        if len(s) > 2 and rng.random() < p_drop:
            s.discard(g[rng.randrange(len(g))])
        for _ in range(rng.randint(1, 2)):
            s.add(rng.randrange(nspr))  # 1-2 sprinkle predicates
        queries.append(sorted(s))
    return {"public": {"M": M, "weights": weights, "queries": queries}, "hidden": {}}


def make_instances():
    specs = [
        # seed,  M,  T, ngrp, glo, ghi, nspr, p_drop
        (7001, 14, 45, 3, 4, 6, 8, 0.25),
        (7002, 15, 50, 3, 4, 6, 9, 0.25),
        (7003, 16, 55, 4, 4, 6, 9, 0.30),
        (7004, 15, 60, 3, 5, 7, 8, 0.20),
        (7005, 17, 70, 4, 4, 7, 10, 0.30),  # noisier
        (7006, 18, 80, 4, 5, 8, 10, 0.30),  # harder
        (7007, 14, 48, 3, 4, 7, 8, 0.25),
        (7008, 19, 90, 5, 5, 8, 11, 0.35),  # hardest / held-out
        (7009, 16, 52, 4, 4, 6, 9, 0.20),
    ]
    return [_build_instance(*sp) for sp in specs]


def _objective(M, weights, queries, order):
    # rank[a] = position of attribute a in the global order
    rank = [0] * M
    for pos, a in enumerate(order):
        rank[a] = pos
    trie = {}
    hit = 0
    for q in queries:
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
        hit += matched
    return hit


def baseline(inst):
    p = inst["public"]
    M = p["M"]
    obj = _objective(M, p["weights"], p["queries"], list(range(M)))
    return max(obj, 1)  # >=1 by construction (repeated identical checks share)


def score(inst, ans):
    p = inst["public"]
    M = p["M"]
    if not isinstance(ans, dict) or "order" not in ans:
        return False, 0
    order = ans["order"]
    if not isinstance(order, list) or len(order) != M:
        return False, 0
    if not all(isinstance(x, int) for x in order):
        return False, 0
    if sorted(order) != list(range(M)):
        return False, 0
    obj = _objective(M, p["weights"], p["queries"], order)
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
