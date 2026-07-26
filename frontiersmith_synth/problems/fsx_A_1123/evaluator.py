#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_1123 -- "Shepherds of the Drifting Flock"
(family: grounded-laplacian-anchors; format B, quality-metric).

THEME. A flock (graph) drifts unless a handful of nodes are pinned as
"shepherds" (anchors). We measure how well an anchor set S controls the whole
flock via the GROUNDED LAPLACIAN: delete the rows/columns of S from the graph
Laplacian L; the remaining (n-|S|)x(n-|S|) matrix L_g is positive definite
(graph connected, |S|>=1) and its SMALLEST eigenvalue lambda_min(L_g) is the
flock's worst-case restoring rate toward the shepherds -- the slowest-decaying
drift mode. We MAXIMIZE lambda_min(L_g) over choices of k anchors.

MECHANISMS COMPOSED.
  (1) grounded-laplacian     -- the objective itself (defined above).
  (2) resistance-farthest-cover -- lambda_min is throttled by whichever node
      is farthest, in EFFECTIVE-RESISTANCE distance, from the anchor set (a
      long pendant tendril's tip is nearly invisible to a purely
      degree/centrality choice of anchors, yet dominates lambda_min).
  (3) non-submodular-pairing -- lambda_min(L_g) is NOT submodular in S: on the
      planted instances, a specific PAIR of anchors placed at once yields a
      strictly larger lambda_min than anything reachable by sequentially
      adding one best-marginal-gain node at a time (the classical
      "additive-gain" leader-selection greedy from the control-theory
      literature). A joint / coordinated search is required.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "k": K (int),
             "edges": [[u,v], ...],     # undirected, simple, connected graph
             "seed": int}
  stdout: ONE JSON object:
            {"anchors": [a_0, ..., a_{K-1}]}   # K distinct node ids in [0,N)

  A submission is VALID iff `anchors` is a list of exactly K integers, all in
  [0,N), all distinct.  Invalid output, a crash, a timeout, or non-JSON output
  makes that instance score 0.0.

SCORING (deterministic; no wall-time). For each instance the evaluator
computes, itself, TWO references never sent to the candidate:
  q_rand   = lambda_min(L_g) for a fixed SEEDED-RANDOM anchor set (a weak,
             topology-blind baseline -- reproducible from `seed` alone).
  q_ref    = lambda_min(L_g) for the BEST of {q_rand, a degree-top-k anchor
             set, and a resistance-farthest-cover construction followed by
             TWO rounds of local pairwise-swap refinement}. q_ref is built to
             weakly DOMINATE the `strong` reference solution (which runs the
             identical construction but stops after ONE refinement round), so
             q_ref >= every reference tier's score by construction --
             guaranteeing headroom above `strong`.
and normalizes affinely:
    r = clamp(0.1 + 0.8 * (q_cand - q_rand) / max(1e-9, q_ref - q_rand), 0, 1)
Matching the random baseline scores ~0.1; matching the internal near-best
reference scores ~0.9 (never 1.0, by the domination construction above) --
a genuinely better joint placement than our reference can still score higher.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. All references
(q_rand, q_ref) are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import numpy as np
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


def _seeded_permutation(seed, n):
    """Deterministic Fisher-Yates shuffle of range(n) using the LCG above."""
    ni = _rng(seed)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = ni(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


# ----------------------------- graph builders -------------------------------
def _add_edge(edges, seen, u, v):
    if u == v:
        return
    key = (u, v) if u < v else (v, u)
    if key in seen:
        return
    seen.add(key)
    edges.append(list(key))


def _attach_tendril(edges, seen, root, start_id, length):
    """Path of `length` NEW nodes hanging off `root`. Returns next free id."""
    prev = root
    nid = start_id
    for _ in range(length):
        _add_edge(edges, seen, prev, nid)
        prev = nid
        nid += 1
    return nid


def _build_twin_tendril(name, seed, hub_gap, tendril_lens_a, tendril_lens_b, bridge_len):
    """Two hub centers (ids 0,1) each sprouting several tendrils, ASSIGNED
    SHORTEST-FIRST so ascending-id tie-breaks land on useless short tendrils
    (this is what stress-tests a degree/marginal-gain style search), joined
    by a bridge path. `tendril_lens_a/b` sorted ascending by caller."""
    edges, seen = [], set()
    hubA, hubB = 0, 1
    nid = 2
    for L in sorted(tendril_lens_a):
        nid = _attach_tendril(edges, seen, hubA, nid, L)
    for L in sorted(tendril_lens_b):
        nid = _attach_tendril(edges, seen, hubB, nid, L)
    # bridge path hubA -> hubB
    prev = hubA
    for _ in range(bridge_len):
        _add_edge(edges, seen, prev, nid)
        prev = nid
        nid += 1
    _add_edge(edges, seen, prev, hubB)
    n = nid
    k = 2 + max(1, (len(tendril_lens_a) + len(tendril_lens_b)) // 3)
    return {"name": name, "n": n, "k": k, "edges": edges, "seed": seed}


def _build_dumbbell_asym(name, seed, clique_size, tendril_lens_b, bridge_len):
    """A dense clique A (all high, near-tied degree) bridged to a lone hub B
    that carries several tendrils.  Node 0 is the clique member touching the
    bridge (highest degree overall); the rest of the clique ties just below
    it -- a pure degree ranking spends its whole budget inside the clique and
    never reaches hub B's side at all."""
    edges, seen = [], set()
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            _add_edge(edges, seen, i, j)
    nid = clique_size
    prev = 0
    for _ in range(bridge_len):
        _add_edge(edges, seen, prev, nid)
        prev = nid
        nid += 1
    hubB = nid
    nid += 1
    _add_edge(edges, seen, prev, hubB)
    for L in sorted(tendril_lens_b):
        nid = _attach_tendril(edges, seen, hubB, nid, L)
    n = nid
    k = 2
    return {"name": name, "n": n, "k": k, "edges": edges, "seed": seed}


def _build_chain_hubs(name, seed, hub_tendrils, bridge_len):
    """A chain of len(hub_tendrils) hub centers (ids 0..H-1) joined by bridge
    paths, each sprouting its own tendrils (shortest-first id assignment).
    Budget = H+1: one anchor per hub plus exactly ONE bonus anchor, which
    must go to the single globally longest tendril across the WHOLE chain --
    a joint, cross-branch decision, not a local ascending-id tie-break."""
    edges, seen = [], set()
    H = len(hub_tendrils)
    nid = H
    for h in range(H):
        for L in sorted(hub_tendrils[h]):
            nid = _attach_tendril(edges, seen, h, nid, L)
    for h in range(H - 1):
        prev = h
        for _ in range(bridge_len):
            _add_edge(edges, seen, prev, nid)
            prev = nid
            nid += 1
        _add_edge(edges, seen, prev, h + 1)
    n = nid
    k = H + 1
    return {"name": name, "n": n, "k": k, "edges": edges, "seed": seed}


def _build_generic_sparse(name, seed, n, k, extra_edge_frac):
    """A random connected sparse graph: a random spanning tree (via a random
    permutation, each new node attaches to a uniformly random earlier node)
    plus a modest number of extra random edges. No planted trap -- exercises
    generalization of whatever strategy the candidate uses."""
    ni = _rng(seed)
    edges, seen = [], set()
    order = list(range(n))
    for i in range(1, n):
        parent = ni(0, i - 1)
        _add_edge(edges, seen, order[i], order[parent])
    extra = int(n * extra_edge_frac)
    tries = 0
    while len(edges) < (n - 1) + extra and tries < extra * 20 + 50:
        u = ni(0, n - 1)
        v = ni(0, n - 1)
        _add_edge(edges, seen, u, v)
        tries += 1
    return {"name": name, "n": n, "k": k, "edges": edges, "seed": seed}


def _build_instances():
    insts = []
    # -- planted traps (resistance-farthest-cover / non-submodular-pairing) --
    insts.append(_build_twin_tendril("twin_tendril_1", 4001, 3, [4, 5, 6], [4, 5, 17], 6))
    insts.append(_build_twin_tendril("twin_tendril_2", 4002, 3, [3, 6, 8], [3, 3, 20], 5))
    insts.append(_build_dumbbell_asym("dumbbell_asym_1", 4011, 6, [3, 3, 16], 9))
    insts.append(_build_dumbbell_asym("dumbbell_asym_2", 4012, 7, [4, 20], 11))
    insts.append(_build_chain_hubs("chain_hubs_1", 4021, [[3, 4], [3, 3], [4, 18]], 4))
    # -- generic / generalization holdout (no planted trap) --
    insts.append(_build_generic_sparse("generic_1", 5001, 30, 3, 0.15))
    insts.append(_build_generic_sparse("generic_2", 5002, 40, 4, 0.20))
    insts.append(_build_generic_sparse("generic_3", 5003, 55, 4, 0.12))
    insts.append(_build_generic_sparse("generic_4", 5004, 65, 5, 0.18))
    insts.append(_build_generic_sparse("generic_5", 5005, 45, 3, 0.35))
    return insts


# ----------------------------- linear algebra -------------------------------
def _laplacian(n, edges):
    L = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= 1.0
        L[v, u] -= 1.0
    return L


def _lambda_min_grounded(L, anchors):
    n = L.shape[0]
    keep = [i for i in range(n) if i not in anchors]
    if not keep:
        return 0.0
    Lg = L[np.ix_(keep, keep)]
    eigs = np.linalg.eigvalsh(Lg)
    return float(eigs[0])


def _pinv_laplacian(L):
    return np.linalg.pinv(L, hermitian=True)


def _resist(Lp, i, j):
    return float(Lp[i, i] + Lp[j, j] - 2.0 * Lp[i, j])


# ----------------------------- reference constructions -----------------------
def _random_anchors(seed, n, k):
    return _seeded_permutation(seed, n)[:k]


def _degree_topk(n, edges, k):
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    order = sorted(range(n), key=lambda i: (-deg[i], i))
    return order[:k]


def _farthest_point_init(Lp, n, k, start):
    anchors = [start]
    while len(anchors) < k:
        best_v, best_d = -1, -1.0
        for v in range(n):
            if v in anchors:
                continue
            d = min(_resist(Lp, v, s) for s in anchors)
            if d > best_d:
                best_d, best_v = d, v
        anchors.append(best_v)
    return anchors


def _swap_pool(Lp, n, anchors, pool_size):
    dists = []
    for v in range(n):
        if v in anchors:
            continue
        d = min(_resist(Lp, v, s) for s in anchors)
        dists.append((d, v))
    dists.sort(reverse=True)
    return [v for _, v in dists[:pool_size]]


def _marginal_gain_init(L, n, k):
    """Sequential marginal-gain (additive-gain) construction -- IDENTICAL
    procedure to solutions/greedy.py. Used as one of the oracle's starting
    points so q_ref weakly dominates the `greedy` tier too."""
    anchors = []
    for _ in range(k):
        best_v, best_obj = -1, -1.0
        for v in range(n):
            if v in anchors:
                continue
            obj = _lambda_min_grounded(L, anchors + [v])
            if obj > best_obj:
                best_obj, best_v = obj, v
        anchors.append(best_v)
    return anchors


def _refine_rounds(L, Lp, n, anchors, rounds, pool_size=20):
    anchors = list(anchors)
    cur = _lambda_min_grounded(L, anchors)
    for _ in range(rounds):
        pool = _swap_pool(Lp, n, anchors, pool_size)
        best_gain, best_ai, best_v, best_obj = 1e-12, -1, -1, cur
        for ai in range(len(anchors)):
            saved = anchors[ai]
            for v in pool:
                if v in anchors:
                    continue
                anchors[ai] = v
                obj = _lambda_min_grounded(L, anchors)
                if obj - cur > best_gain:
                    best_gain, best_ai, best_v, best_obj = obj - cur, ai, v, obj
            anchors[ai] = saved
        if best_ai < 0:
            break
        anchors[best_ai] = best_v
        cur = best_obj
    return anchors, cur


def _reference_values(inst):
    """(q_rand, q_ref) -- both computed ONLY from the instance, never from a
    candidate answer. q_ref weakly DOMINATES both the `greedy` and `strong`
    solution tiers by construction: it refines the SAME two starting
    constructions those tiers use (farthest-point-cover from the top-degree
    node, and sequential marginal-gain) with the SAME pool-selection rule,
    but for TWO refinement rounds instead of one -- round 1 of this refine
    reproduces what `strong`/`greedy` computed, and round 2 can only match or
    improve it (monotonic, never-decreasing local search)."""
    n, k, edges, seed = inst["n"], inst["k"], inst["edges"], inst["seed"]
    L = _laplacian(n, edges)
    Lp = _pinv_laplacian(L)

    rand_anchors = _random_anchors(seed, n, k)
    q_rand = _lambda_min_grounded(L, rand_anchors)

    deg_anchors = _degree_topk(n, edges, k)
    q_deg = _lambda_min_grounded(L, deg_anchors)

    start = deg_anchors[0]
    cover_init = _farthest_point_init(Lp, n, k, start)
    _, q_cover2 = _refine_rounds(L, Lp, n, cover_init, rounds=2, pool_size=20)

    gain_init = _marginal_gain_init(L, n, k)
    _, q_gain2 = _refine_rounds(L, Lp, n, gain_init, rounds=2, pool_size=20)

    q_ref = max(q_rand, q_deg, q_cover2, q_gain2)
    return q_rand, q_ref


# ----------------------------- validation -----------------------------------
def _validate_answer(n, k, answer):
    if not isinstance(answer, dict):
        return None
    anchors = answer.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != k:
        return None
    seen = set()
    out = []
    for a in anchors:
        if isinstance(a, bool) or not isinstance(a, int):
            return None
        if a < 0 or a >= n or a in seen:
            return None
        seen.add(a)
        out.append(a)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        n, k, edges = inst["n"], inst["k"], inst["edges"]
        q_rand, q_ref = _reference_values(inst)
        denom = max(1e-9, q_ref - q_rand)

        public = {"name": inst["name"], "n": n, "k": k,
                  "edges": [list(e) for e in edges], "seed": inst["seed"]}
        ans, st = isorun.run_candidate(cand, public, timeout=15)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            anchors = _validate_answer(n, k, ans)
        except Exception:
            anchors = None
        if anchors is None:
            vec.append(0.0)
            continue
        L = _laplacian(n, edges)
        try:
            q_cand = _lambda_min_grounded(L, anchors)
        except Exception:
            vec.append(0.0)
            continue
        if not (q_cand == q_cand) or q_cand in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = 0.1 + 0.8 * (q_cand - q_rand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
