#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0907 -- "Riverdelta Levee Network: Budgeted Gauge Recalibration"
(family: gauss-southwell-block-descent; format B, quality-metric).

THEME.  A river-delta flood-control authority operates N water-level gauges.  Each
gauge i has a private drift correction it wants to settle toward its own calibration
target c_i (weighted by confidence a_i), but some gauges are hydraulically COUPLED
through shared sluice channels: a coupling edge (i, j, w) means gauges i and j must
be reconciled against each other with stiffness w, because water actually flowing
through the shared channel makes their readings interact.  The coupling GRAPH is
heterogeneous and only partially resembles what the raw index order suggests: most
gauges are hydraulically isolated (their own optimum never changes once corrected),
but a minority sit inside one or two tightly-coupled channel clusters that need
several rounds of back-and-forth reconciliation before they settle.

Formally this is separable-plus-coupling quadratic minimization:

    f(x) = sum_i a_i * (x_i - c_i)^2  +  sum_{(i,j) in E} w_ij * (x_i - x_j)^2

You are given a FIXED BUDGET of K coordinate-update steps.  Each step names one
gauge index i; that gauge is reset EXACTLY to its own coordinate optimum given the
CURRENT values of every other gauge (closed form, see below) -- this can only ever
decrease or hold f, never increase it.  Your program chooses the length-K sequence
of gauge indices (repeats allowed) that will be applied, in order, starting from a
given initial vector x0.  The evaluator replays your exact sequence and scores the
final objective.  With K typically only slightly above N, WHICH gauges get the
"extra" visits (beyond one each) decides almost the entire outcome: spending them on
already-settled isolated gauges is pure waste, while pouring them into the stiff
coupled cluster(s) can close most of the remaining gap.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N, "K": K,
             "a": [a_0..a_{N-1}], "c": [c_0..c_{N-1}], "x0": [x0_0..x0_{N-1}],
             "edges": [[i, j, w], ...]}    # undirected coupling edges, w > 0
  stdout: ONE JSON object:
            {"order": [i_0, i_1, ..., i_{K-1}]}   # exactly K gauge indices, each
                                                    # an integer in [0, N)

  The per-step closed-form coordinate optimum used by the replay (S_i = sum of w
  over i's incident edges, nbrs(i) = i's coupled neighbours):
      x_i <-  (a_i * c_i + sum_{j in nbrs(i)} w_ij * x_j) / (a_i + S_i)

  A layout is VALID iff `order` is a list of exactly K integers, each in [0, N).
  Invalid output (wrong length, out-of-range / non-integer index, a crash, a
  timeout, or non-JSON output) makes that instance score 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes, itself:
    f_lb   = the TRUE unconstrained global minimum of f (exact linear solve) --
             a generally-unreachable ideal within a finite budget on the stiff
             clusters,
    f_base = f reached by the trivial construction "one full pass over every
             gauge in index order (0..N-1), then waste the remaining K-N
             surplus steps re-updating gauge 0" (every gauge gets its
             mandatory single visit, but the surplus is spent with zero
             regard for the coupling structure -- this isolates scoring on
             exactly how the SURPLUS budget is spent),
    f_cand = f reached by REPLAYING the candidate's own `order`,
and normalizes with an affine anchor (weak baseline -> 0.1, true optimum -> 1.0):

    r = clamp( 0.1 + 0.9 * (f_base - f_cand) / max(1e-9, f_base - f_lb), 0, 1 )

Matching the "always gauge 0" baseline scores ~0.1; reaching the true optimum
scores 1.0; doing worse than the baseline scores below 0.1.  Because several
instances plant a stiff, poorly-conditioned coupled cluster that cannot be fully
settled within budget K, even an excellent allocator stays comfortably below 1.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  f_lb / f_base /
the replay are all computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, random
import isorun

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


# ----------------------------- instance construction ------------------------
def _weight(rng, difficulty):
    """Sample one coupling weight; 'hard' clusters skew strong (slow, ill-
    conditioned Gauss-Seidel convergence), 'easy' clusters skew weak (fast)."""
    if difficulty == "hard":
        return rng.uniform(30.0, 120.0) if rng.random() < 0.75 else rng.uniform(0.4, 3.0)
    if difficulty == "medium":
        return rng.uniform(8.0, 25.0) if rng.random() < 0.50 else rng.uniform(0.4, 3.0)
    return rng.uniform(8.0, 18.0) if rng.random() < 0.15 else rng.uniform(0.4, 3.0)  # easy


def _cluster_diag(rng, difficulty):
    """Diagonal weight a_i for a cluster node -- kept small for 'hard' clusters
    so the diagonal barely dominates the coupling row-sum (near-degenerate
    diagonal dominance -> slow per-visit convergence)."""
    if difficulty == "hard":
        return rng.uniform(0.02, 0.12)
    if difficulty == "medium":
        return rng.uniform(0.4, 1.2)
    return rng.uniform(0.8, 2.5)


def _build_cluster_edges(rng, nodes, difficulty):
    """'hard' clusters are a CHAIN (path graph) over `nodes` in a RANDOM
    logical order -- the node ids that are actually adjacent along the chain
    give no hint from their numeric index. A chain has an intrinsic
    propagation delay: a coordinate's true optimum depends on information
    that must relay hop-by-hop from one end, so no ordering can shortcut past
    O(length) sequential, topology-respecting visits; an index-oblivious
    sweep (which touches chain positions in an essentially random sequence)
    propagates very little per pass, while a solver that diagnoses the chain
    and visits it end-to-end propagates almost optimally.
    'easy'/'medium' clusters use a random spanning tree plus a few extra
    cycle edges -- a richer but structurally forgiving topology.
    Returns list of (i, j, w)."""
    order = list(nodes)
    rng.shuffle(order)
    edges = []
    if difficulty == "hard":
        for k in range(len(order) - 1):
            i, j = order[k], order[k + 1]
            edges.append((min(i, j), max(i, j), _weight(rng, difficulty)))
        return edges
    seen = set()
    for k in range(1, len(order)):
        j = order[k]
        i = order[rng.randrange(k)]      # attach to a random earlier node
        w = _weight(rng, difficulty)
        edges.append((min(i, j), max(i, j), w))
        seen.add((min(i, j), max(i, j)))
    extra = max(1, len(order) // 3)
    tries = 0
    while extra > 0 and tries < 50:
        tries += 1
        i, j = rng.sample(order, 2)
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        edges.append((key[0], key[1], _weight(rng, difficulty)))
        extra -= 1
    return edges


def _build_instance(seed, n, cluster_specs, extra, name):
    """cluster_specs: list of (size, difficulty). Node 0 is always an isolated
    singleton (the trivial baseline's fixed target)."""
    rng = random.Random(seed)
    cluster_total = sum(sz for sz, _ in cluster_specs)
    assert cluster_total < n - 1, "need spare singletons besides node 0"
    pool = rng.sample(range(1, n), cluster_total)  # random node ids -> clusters
    rng.shuffle(pool)
    groups = []
    p = 0
    for sz, diff in cluster_specs:
        groups.append((pool[p:p + sz], diff))
        p += sz

    a = [0.0] * n
    c = [rng.uniform(-12.0, 12.0) for _ in range(n)]
    x0 = [rng.uniform(-12.0, 12.0) for _ in range(n)]
    edges = []
    cluster_ids = set()
    for nodes, diff in groups:
        for i in nodes:
            a[i] = _cluster_diag(rng, diff)
            cluster_ids.add(i)
        edges.extend(_build_cluster_edges(rng, nodes, diff))

    for i in range(n):
        if i not in cluster_ids:
            a[i] = rng.uniform(0.5, 2.5)   # singleton gauges (incl. node 0)

    K = n + extra
    return {"name": name, "n": n, "K": K, "a": a, "c": c, "x0": x0,
            "edges": [[i, j, round(w, 6)] for (i, j, w) in edges]}


def _build_instances():
    specs = [
        # (seed, n, cluster_specs, extra, name)
        (30101, 16, [(3, "easy")], 5, "delta101"),
        (30102, 18, [(4, "easy")], 5, "delta102"),
        (30103, 20, [(5, "medium")], 6, "delta103"),
        (30104, 20, [(6, "hard")], 8, "delta104"),
        (30105, 22, [(7, "hard")], 9, "delta105"),
        (30106, 24, [(8, "hard")], 10, "delta106"),
        (30107, 20, [(4, "hard"), (4, "medium")], 9, "delta107"),
        (30108, 18, [(5, "hard")], 7, "delta108"),
        # harder / larger held-out instances
        (30109, 30, [(11, "hard")], 13, "delta109"),
        (30110, 32, [(6, "hard"), (6, "hard")], 12, "delta110"),
    ]
    return [_build_instance(seed, n, cs, ex, name) for seed, n, cs, ex, name in specs]


# ----------------------------- shared math -----------------------------
def _adjacency(n, edges):
    nbrs = [[] for _ in range(n)]
    S = [0.0] * n
    for i, j, w in edges:
        nbrs[i].append((j, w))
        nbrs[j].append((i, w))
        S[i] += w
        S[j] += w
    return nbrs, S


def _fval(a, c, edges, x):
    total = sum(a[i] * (x[i] - c[i]) ** 2 for i in range(len(a)))
    for i, j, w in edges:
        total += w * (x[i] - x[j]) ** 2
    return total


def _simulate(n, a, c, nbrs, S, x0, order):
    x = list(x0)
    for i in order:
        num = a[i] * c[i]
        for j, w in nbrs[i]:
            num += w * x[j]
        x[i] = num / (a[i] + S[i])
    return x


def _solve_lb(n, a, c, edges, nbrs, S):
    """Exact global minimum of f via the linear system grad f = 0."""
    if _np is not None:
        M = _np.zeros((n, n), dtype=float)
        b = _np.zeros(n, dtype=float)
        for i in range(n):
            M[i, i] = a[i] + S[i]
            b[i] = a[i] * c[i]
        for i, j, w in edges:
            M[i, j] -= w
            M[j, i] -= w
        xstar = _np.linalg.solve(M, b)
        return list(xstar)
    # pure-python Gauss-Seidel fallback (many sweeps -> converges for this SPD system)
    x = list(c)
    for _ in range(4000):
        for i in range(n):
            num = a[i] * c[i]
            for j, w in nbrs[i]:
                num += w * x[j]
            x[i] = num / (a[i] + S[i])
    return x


def _prep(inst):
    n = inst["n"]; a = inst["a"]; c = inst["c"]
    edges = [(int(i), int(j), float(w)) for i, j, w in inst["edges"]]
    nbrs, S = _adjacency(n, edges)
    return n, a, c, edges, nbrs, S


def _base_order(n, K):
    """One deterministic full pass over every gauge (index order), then the
    remaining surplus budget is wasted on an already-converged gauge (index 0)
    -- i.e. every gauge gets its mandatory single visit, but the surplus is
    spent with zero regard for the coupling structure. This isolates the
    score on exactly the thing under test: how the SURPLUS budget is spent."""
    base = list(range(n))
    if K > n:
        base += [0] * (K - n)
    return base[:K]


def baseline_and_lb(inst):
    n, a, c, edges, nbrs, S = _prep(inst)
    x_base = _simulate(n, a, c, nbrs, S, inst["x0"], _base_order(n, inst["K"]))
    f_base = _fval(a, c, edges, x_base)
    x_lb = _solve_lb(n, a, c, edges, nbrs, S)
    f_lb = _fval(a, c, edges, x_lb)
    return f_base, f_lb


def score(inst, answer):
    """Validate + replay the candidate's order. Return (ok, f_cand)."""
    if not isinstance(answer, dict):
        return False, None
    order = answer.get("order")
    n = inst["n"]; K = inst["K"]
    if not isinstance(order, list) or len(order) != K:
        return False, None
    clean = []
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int):
            return False, None
        if v < 0 or v >= n:
            return False, None
        clean.append(v)
    n_, a, c, edges, nbrs, S = _prep(inst)
    x_final = _simulate(n, a, c, nbrs, S, inst["x0"], clean)
    f_cand = _fval(a, c, edges, x_final)
    if f_cand != f_cand or f_cand in (float("inf"), float("-inf")):
        return False, None
    return True, f_cand


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        f_base, f_lb = baseline_and_lb(inst)
        denom = f_base - f_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "n": inst["n"], "K": inst["K"],
                  "a": list(inst["a"]), "c": list(inst["c"]), "x0": list(inst["x0"]),
                  "edges": [list(e) for e in inst["edges"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, f_cand = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (f_base - f_cand) / denom
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
