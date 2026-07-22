#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0824 -- "One Question, Four Shuffled Haystacks"
(family: probe-tree-shared-across-priors; format B, quality-metric).

THEME.  There are N labeled items (a fixed public universe, ids 0..N-1). A
target item is hidden, drawn from one of K=4 fixed SCENARIO POOLS -- each
pool is its own probability distribution over the SAME N items, built by
clumping most of that pool's mass onto a small "home" CLUSTER of items
scattered arbitrarily through the id space (a different, differently
shuffled clumping per pool), with a thin residue spread over everything
else. Your program is a diagnostic strategy: an adaptive binary decision
tree of subset PROBES ("is the target in S?") that must correctly identify
the target no matter which pool it actually came from (the tree is built
ONCE, from the public pool descriptions, before any probing happens --
there is no per-probe interaction; you never get told which pool is live).

SCORING (deterministic; no wall-time). For every one of the N items i, the
evaluator walks your tree using the TRUTHFUL membership answers for i and
records probes_to_identify(i) (and requires the tree to correctly name i at
the leaf it reaches -- any wrong / stuck / malformed traversal invalidates
the WHOLE instance). Each pool's quality is then its own PROBES MEAN:
   mean_k = sum_i pool_k[i] * probes_to_identify(i)
and your instance objective is the MAXIMUM over the 4 pools of mean_k (you
must minimize this -- a strategy that is excellent for three pools but
terrible for the pool whose clumping it mishandles is scored on that worst
pool, not the average). This is the trap: a tree tuned to split the POOLED
(mixture) distribution as evenly as possible is optimal on average, but
some individual pool's clumping straddles that split unevenly, costing that
pool extra probes it didn't need to pay.

Reference anchors, computed directly by THIS evaluator (never from a
candidate): `weak` = the objective of a fixed "scan items in a fixed order,
one at a time" reference tree (no adaptivity, ignores the pools entirely);
`lb` = the largest per-pool Shannon entropy max_k H(pool_k) (bits), an
information-theoretic floor: no tree, however constructed, can beat this
for pool k's own mean (Kraft's inequality on the depths of a prefix code).
Both pools' worth of structure make lb usually far below anything a single
shared tree can reach across ALL 4 pools at once, which is exactly why
`lb` never saturates -- it leaves headroom above even a very good policy.

    r = clamp( 0.1 + 0.9 * (weak - obj) / max(weak - lb, 1e-6), 0, 1 )

ISOLATION. Your program is called EXACTLY ONCE per instance, in a fresh
sandboxed subprocess via `isorun.run_candidate`; it only ever sees that
instance's public pool weights. The scoring walk, the weak reference, and
the entropy floor are all computed only in this parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun

CALL_TIMEOUT = 20          # seconds for the single per-instance isorun call
MAX_ANSWER_CHARS = 2_000_000


# ----------------------------- deterministic instance family ---------------
def _shuffled_clusters(n, sizes, seed):
    """Deterministic (stdlib random, fixed seed) partition of 0..n-1 into
    len(sizes) clusters of the given sizes, positions scrambled through the
    id space -- a different 'shuffled haystack' arrangement per seed."""
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    clusters, pos = [], 0
    for sz in sizes:
        clusters.append(sorted(ids[pos:pos + sz]))
        pos += sz
    return clusters


def _make_pools(n, clusters, heavy_masses):
    """K pools over the same n items: pool k puts `heavy_masses[k]` total
    mass uniformly over its own cluster, the remaining mass uniformly over
    every other item (clusters partition the full item universe)."""
    pools = []
    for k, c in enumerate(clusters):
        w = [0.0] * n
        other = [i for i in range(n) if i not in c]
        heavy_each = heavy_masses[k] / len(c)
        resid_each = (1.0 - heavy_masses[k]) / max(len(other), 1)
        for i in c:
            w[i] = heavy_each
        for i in other:
            w[i] += resid_each
        pools.append(w)
    return pools


# name, cluster sizes (K=4 pools each with its own home cluster), each pool's
# heavy mass on its own cluster, and the shuffle seed scattering the clusters
# through the id space. All verified offline to satisfy the acceptance gates.
_SPECS = [
    dict(name="haystack01", sizes=[3, 3, 3, 3], heavy=[0.97, 0.90, 0.70, 0.60], seed=3),
    dict(name="haystack02", sizes=[3, 3, 3, 3], heavy=[0.98, 0.88, 0.68, 0.58], seed=34),
    dict(name="haystack03", sizes=[3, 3, 3, 3], heavy=[0.95, 0.88, 0.72, 0.62], seed=3),
    dict(name="haystack04", sizes=[3, 3, 3, 3], heavy=[0.97, 0.90, 0.70, 0.60], seed=34),
    dict(name="haystack05", sizes=[3, 3, 3, 3], heavy=[0.97, 0.90, 0.70, 0.60], seed=126),
    dict(name="haystack06", sizes=[3, 3, 3, 3], heavy=[0.98, 0.88, 0.68, 0.58], seed=126),
    dict(name="haystack07", sizes=[3, 3, 3, 3], heavy=[0.95, 0.88, 0.72, 0.62], seed=34),
    dict(name="haystack08", sizes=[3, 3, 4, 4], heavy=[0.92, 0.85, 0.75, 0.65], seed=64),
    dict(name="haystack09", sizes=[3, 3, 3, 3], heavy=[0.97, 0.90, 0.70, 0.60], seed=82),
    dict(name="haystack10", sizes=[3, 4, 3, 4], heavy=[0.95, 0.85, 0.70, 0.60], seed=77),
]


def _build_instances():
    out = []
    for spec in _SPECS:
        n = sum(spec["sizes"])
        clusters = _shuffled_clusters(n, spec["sizes"], spec["seed"])
        pools = _make_pools(n, clusters, spec["heavy"])
        out.append({"name": spec["name"], "n": n, "k": len(clusters), "pools": pools})
    return out


# ----------------------------- reference constructions ----------------------
def _entropy(w):
    h = 0.0
    for x in w:
        if x > 1e-15:
            h -= x * math.log2(x)
    return h


def _lower_bound(inst):
    return max(_entropy(p) for p in inst["pools"])


def _weak_tree(n):
    """Fixed 'scan items 0,1,2,... one at a time' reference tree -- no
    adaptivity, completely ignores the pools."""
    def rec(lo, hi):
        if hi - lo == 1:
            return {"guess": lo}
        return {"query": [lo], "yes": {"guess": lo}, "no": rec(lo + 1, hi)}
    return rec(0, n)


# ----------------------------- tree walking / validation --------------------
def _walk(tree, target, n, depth_cap):
    """Walk `tree` for a known target id, using TRUTHFUL membership answers.
    Returns (probes, guessed_id). Raises ValueError on any malformed node."""
    depth = 0
    cur = tree
    while True:
        if not isinstance(cur, dict):
            raise ValueError("node not an object")
        if "guess" in cur:
            g = cur["guess"]
            if isinstance(g, bool) or not isinstance(g, int) or not (0 <= g < n):
                raise ValueError("bad guess")
            return depth, g
        if "query" not in cur:
            raise ValueError("node missing query/guess")
        q = cur["query"]
        if not isinstance(q, list) or not (1 <= len(q) <= n - 1):
            raise ValueError("bad query length")
        qset = set()
        for x in q:
            if isinstance(x, bool) or not isinstance(x, int) or not (0 <= x < n):
                raise ValueError("bad query id")
            qset.add(x)
        if len(qset) != len(q):
            raise ValueError("duplicate ids in query")
        branch = cur.get("yes") if (target in qset) else cur.get("no")
        if not isinstance(branch, dict):
            raise ValueError("missing branch")
        cur = branch
        depth += 1
        if depth > depth_cap:
            raise ValueError("depth cap exceeded")


def _objective(inst, tree):
    """Returns the instance objective (max over pools of the probe-count
    mean) for a validated tree, or None if the tree is invalid/incorrect
    for any item."""
    n = inst["n"]
    depth_cap = n + 4
    depths = [None] * n
    for i in range(n):
        try:
            d, g = _walk(tree, i, n, depth_cap)
        except (ValueError, RecursionError, TypeError, KeyError):
            return None
        if g != i:
            return None
        depths[i] = d
    means = []
    for pool in inst["pools"]:
        tot = sum(pool)
        m = sum(pool[i] * depths[i] for i in range(n)) / tot if tot > 0 else 0.0
        means.append(m)
    return max(means)


# ----------------------------- candidate plumbing ----------------------------
def _public_view(inst):
    return {"n_items": inst["n"], "items": list(range(inst["n"])),
            "n_pools": inst["k"],
            "pools": [{"pool": k, "weights": [round(x, 8) for x in w]}
                      for k, w in enumerate(inst["pools"])]}


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        weak_obj = _objective(inst, _weak_tree(inst["n"]))
        if weak_obj is None:
            weak_obj = float(inst["n"])  # should never happen; safe fallback
        lb = _lower_bound(inst)

        public = _public_view(inst)
        answer, status = isorun.run_candidate(cand, public, timeout=CALL_TIMEOUT)
        if status != "OK" or not isinstance(answer, dict):
            vec.append(0.0)
            continue
        try:
            if len(json.dumps(answer)) > MAX_ANSWER_CHARS:
                vec.append(0.0)
                continue
        except (TypeError, ValueError):
            vec.append(0.0)
            continue
        tree = answer.get("tree")
        obj = _objective(inst, tree) if isinstance(tree, dict) else None
        if obj is None:
            vec.append(0.0)
            continue
        denom = max(weak_obj - lb, 1e-6)
        r = 0.1 + 0.9 * (weak_obj - obj) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
