#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0158 -- "Watchtower Scan-Directive Prefix Cache".

Family: offline-decision-policy (Frontier-CS llm_sql prefix-cache anchor), skinned
as a network of forest-fire watchtowers. Each day the fire-control center issues a
LOG of scan directives (in a fixed, given order). A directive is a SET of sensor
check-clauses drawn from a shared catalog (thermal, smoke, wind, humidity, camera,
lightning, ...). The tower's inference server evaluates a directive as an ORDERED
sequence of clauses and memoizes results by PREFIX (a prefix KV-cache / trie): when
a directive's leading clauses -- in the emitted order -- exactly match a prefix that
was already computed by an EARLIER directive in the log, that leading work is served
from cache (a HIT) instead of recomputed (a MISS). Each clause carries a compute
weight (token cost).

The center gets to fix ONE global canonical CLAUSE ORDER (a permutation of the whole
catalog). Every directive is emitted with its own clauses sorted by this canonical
order, then run through the prefix cache. The candidate must choose the canonical
order that MAXIMIZES the weighted prefix-cache hit rate over the day's log. This is
the offline-decision / column-reordering problem: marginal-frequency ordering is a
strong heuristic but not optimal because prefix sharing depends on clause
co-occurrence structure, not just how often each clause appears -- so there is real
headroom for co-occurrence-aware local search, and no easy optimum.

The candidate is UNTRUSTED: it is run as an ISOLATED stdin->stdout subprocess via
`isorun`, so it only ever sees the public instance and can never reach the
evaluator's frames / cache simulator / scorer. All scoring is deterministic (seeded
instance generation, pure integer arithmetic; no wall-time).

Scoring (maximization; higher hit rate is better):
  obj(order) = weighted prefix-cache hit rate of `order` over the log, in [0,1)
  baseline b = obj(identity order = [0,1,...,C-1])   (computed by the evaluator)
  For a valid permutation with objective obj:  r = min(1, 0.1 * obj / b)
  -> the identity order maps to exactly 0.1; an order whose hit rate is k times the
     identity baseline maps to min(1, 0.1*k). A non-permutation / malformed answer
     scores 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

MASK = (1 << 64) - 1


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & MASK

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        return lo + (state >> 17) % (hi - lo + 1)
    return nxt


def _pick_k(r, pool, k):
    pool = list(pool)
    res = []
    k = min(k, len(pool))
    for _ in range(k):
        idx = r(0, len(pool) - 1)
        res.append(pool.pop(idx))
    return res


def _wpick(r, weights):
    tot = sum(weights)
    x = r(0, tot - 1)
    c = 0
    for i, w in enumerate(weights):
        c += w
        if x < c:
            return i
    return len(weights) - 1


# ----------------------------- prefix-cache simulator ----------------------
def hit_rate(directives, weights, order):
    """Weighted prefix-cache hit rate of emitting each directive's clauses in the
    canonical `order`, processing directives in log order over a shared prefix
    (trie) cache that is prefix-closed by construction."""
    pos = [0] * len(order)
    for i, c in enumerate(order):
        pos[c] = i
    cache = set()
    total = 0
    hit = 0
    for d in directives:
        seq = sorted(d, key=lambda c: pos[c])
        prefixes = []
        t = ()
        for c in seq:
            t = t + (c,)
            prefixes.append(t)
        p = 0
        for k in range(len(prefixes)):
            if prefixes[k] in cache:
                p = k + 1
            else:
                break
        for i, c in enumerate(seq):
            w = weights[c]
            total += w
            if i < p:
                hit += w
        for t in prefixes:
            cache.add(t)
    return hit / total if total > 0 else 0.0


# ----------------------------- instance family -----------------------------
def make_instances():
    """Deterministic, seeded. Returns list of {'public':..., 'hidden':{}}.
    The catalog splits into two roles:
      * HUB clauses -- the two HIGHEST-index clauses (e.g. regional-relay + master
        check). They are near-universal (present in most directives) and carry high
        compute weight.
      * SPECIFIC clauses -- the remaining low-index sensors. They are rare and
        diverse, and are grouped into T co-occurrence CLUSTERS (a directive draws
        mostly from one cluster) with per-clause frequencies that vary WITHIN a
        cluster.
    Because the hubs have the highest indices, the identity (catalog) order emits
    them LAST, so their (heavy, near-universal) work is almost never served from
    cache -> the identity baseline is poor. A good order hoists the hubs to the
    FRONT so nearly every directive shares them; a co-occurrence-aware order further
    arranges each cluster's clauses to extend the shared prefix, which pure marginal
    frequency does not fully capture."""
    specs = [
        # (seed, C, T, Q)
        (10101, 8, 3, 44), (10102, 9, 3, 50), (10103, 9, 3, 46),
        (10104, 10, 3, 52), (10105, 8, 3, 48), (10106, 10, 4, 56),
        # larger / held-out instances
        (10107, 11, 4, 62), (10108, 10, 4, 60), (10109, 11, 4, 66),
        (10110, 11, 4, 70), (10111, 12, 4, 72), (10112, 12, 4, 76),
    ]
    out = []
    for seed, C, T, Q in specs:
        r = _rng(seed)
        weights = [r(1, 2) for _ in range(C)]
        h1, h2 = C - 1, C - 2                       # the two hub clauses
        weights[h1] = r(9, 12)
        weights[h2] = r(6, 9)
        hub_prob = {h1: 970, h2: 900}               # per-mille presence
        specifics = list(range(C - 2))
        # round-robin the specifics into T co-occurrence clusters
        clusters = [[] for _ in range(T)]
        for idx, c in enumerate(specifics):
            clusters[idx % T].append(c)
        # per-specific within-cluster inclusion prob (varies -> marginal freqs differ)
        incl = {}
        for cl in clusters:
            for c in cl:
                incl[c] = r(250, 550)
        cluster_w = [r(1, 5) for _ in range(T)]
        noise_prob = 160                            # cross-cluster diversity noise
        directives = []
        for _ in range(Q):
            d = set()
            if r(0, 999) < hub_prob[h1]:
                d.add(h1)
            if r(0, 999) < hub_prob[h2]:
                d.add(h2)
            t = _wpick(r, cluster_w)
            for c in clusters[t]:
                if r(0, 999) < incl[c]:
                    d.add(c)
            for c in specifics:
                if r(0, 999) < noise_prob:
                    d.add(c)
            if len(d) < 2:
                d.add(h1)
                d.add(clusters[t][0] if clusters[t] else h2)
            directives.append(sorted(d))
        # force a positive baseline floor: two exact duplicates
        directives.append(list(directives[0]))
        directives.append(list(directives[len(directives) // 2]))
        public = {"n_clauses": C, "weights": weights, "directives": directives}
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
def baseline(inst):
    p = inst["public"]
    C = p["n_clauses"]
    return hit_rate(p["directives"], p["weights"], list(range(C)))


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    p = inst["public"]
    C = p["n_clauses"]
    if not isinstance(answer, dict):
        return False, None
    order = answer.get("order", None)
    if not isinstance(order, list) or len(order) != C:
        return False, None
    try:
        order = [int(x) for x in order]
    except (TypeError, ValueError):
        return False, None
    if sorted(order) != list(range(C)):
        return False, None
    obj = hit_rate(p["directives"], p["weights"], order)
    if obj != obj or obj < 0.0:
        return False, None
    return True, obj


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
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
            ok, obj = False, None
        if not ok or obj is None:
            vec.append(0.0)
            continue
        b = baseline(inst)
        if b <= 0:
            vec.append(0.0)
            continue
        r = min(1.0, 0.1 * obj / b)
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
