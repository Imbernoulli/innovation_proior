#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0398 -- "Museum Gallery Tour: Recovering the Visitor-Flow
Causal Map" (family: causal-discovery; eval_form: quality-metric; MLS-Bench causal-* shape).

A museum instruments its galleries with anonymous foot-traffic beacons.  For every
visitor tour it records a BINARY vector: for each gallery, whether that visitor
entered it (1) or not (0).  Whether a visitor steps into one gallery causally
raises the chance they step into certain OTHER galleries downstream (a signpost, a
shared doorway, a "if you liked X you'll like Y" placard).  These influences form a
HIDDEN directed acyclic graph (DAG): each gallery's entry is a NOISY-OR of a small
private "leak" rate plus activation contributed by each upstream gallery the
visitor already entered (a discrete / categorical Bayesian-network structural
model -- the Cancer/Child/Alarm family, not linear-Gaussian).  Only OBSERVATIONAL
tour logs are available -- no interventions, no forced routing.

The candidate designs a CAUSAL-DISCOVERY routine.  It is run as an ISOLATED
subprocess (isorun): it reads ONE JSON "public instance" from stdin (the raw
binary tour matrix + gallery count) and writes ONE JSON answer (a predicted set of
directed edges) to stdout.  It NEVER sees the ground-truth DAG, the topological
order, the noisy-OR weights, the leak rates, or this evaluator's memory.

Public instance JSON (stdin):
    {
      "data":        [[0/1, ...], ...],   # N x d binary tour matrix (fresh copy)
      "n_tours":     int,                 # number of tours N (rows)
      "n_galleries": int,                 # number of galleries d (columns)
      "gallery_names":[str, ...],         # flavor labels for the d galleries
      "seed":        int                  # per-instance seed the candidate MAY use
    }

Answer JSON (stdout) -- ANY of these forms is accepted:
    [[i, j], ...]                         # directed edges  i -> j  (gallery i drives gallery j)
    {"edges": [[i, j], ...]}              # same, wrapped
    {"adjacency": [[0/1, ...], ...]}      # d x d adjacency matrix, A[i][j]=1 means i -> j

Quality is the STRUCTURAL HAMMING DISTANCE (SHD) between the predicted graph and
the hidden ground-truth DAG, recomputed deterministically here.  SHD counts, over
every unordered gallery pair, the edits needed to match truth: a MISSING edge, an
EXTRA edge, or a REVERSED / ambiguous orientation each cost 1.  Lower is better.

The evaluator's own trivial baseline is the EMPTY graph, whose SHD equals the
number of true edges E.  Per instance the normalized score is the minimization
form from the authoring contract:

    r = min( 1.0, 0.1 * SHD_empty / max(SHD_cand, 1e-9) )

so a candidate no better than the empty map maps to ~0.1 and a perfect recovery
(SHD = 0) maps to 1.0.  With only OBSERVATIONAL discrete data, orientation is
under-determined (Markov-equivalence) and finite tours leave dependence estimates
noisy, so even a strong routine keeps headroom below 1.0 -- especially on the
larger, sparser, tour-poorer wings and the held-out galleries.  The final score is
the MEAN of per-instance r over a diverse battery of museums, rewarding a rule that
GENERALIZES.  A crash / timeout / malformed / out-of-range answer scores 0.0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, random
import isorun

VALID_FLOOR = 0.02      # floor for VALID instances (weak-but-valid stays > invalid=0)
CAND_TIMEOUT = 20

_GALLERIES = [
    "Antiquities", "Impressionists", "ModernWing", "Sculpture", "Photography",
    "Textiles", "Armory", "Cartography", "Botanicals", "Ceramics",
    "Manuscripts", "Astronomy", "Numismatics", "Puppetry",
]


# ============================ museum-tour family (instances) ================
def _gen_bn(seed, d, n, edge_prob, leak_lo=0.10, leak_hi=0.18,
            w_lo=0.30, w_hi=0.66, max_parents=4):
    """Generate a discrete noisy-OR Bayesian network on d galleries and draw N
    binary tours.  Edges are built in a topological order then the columns are
    RELABELLED by a random permutation, so raw index order carries NO orientation
    information.  Each gallery b entered with probability

        P(b=1 | active parents S) = 1 - (1 - leak[b]) * prod_{a in S} (1 - w[a,b])

    (noisy-OR).  Source galleries fire only at their small leak rate, so they are
    RARER than their descendants -- an ascending-marginal-frequency ordering weakly
    recovers the topological order (the discrete analogue of equal-variance
    ordering), which a good routine can exploit.  Colliders (a common child of two
    galleries) keep the two parents MARGINALLY independent, so a marginal-dependence
    skeleton does NOT hallucinate a co-parent edge -- but TRANSITIVE dependence along
    a path A->B->C does show up as a false A-C edge unless conditioned away.

    Returns (true_edges, data) where true_edges is a set of (i, j) in DATA-column
    space meaning gallery i -> gallery j, and data is an N x d list of 0/1 ints.
    """
    rng = random.Random(seed)
    # 1) build edges in topological space (topo node a -> topo node b, a < b),
    #    capped in-degree so descendants do not saturate to ~1 (which would erase
    #    the frequency-ordering signal).
    indeg = [0] * d
    topo_edges = []
    for b in range(d):
        for a in range(b):
            if indeg[b] >= max_parents:
                break
            if rng.random() < edge_prob:
                topo_edges.append((a, b))
                indeg[b] += 1
    # guarantee at least 2 edges so the empty-graph baseline is well defined
    tries = 0
    while len(topo_edges) < 2 and tries < 200:
        a = rng.randrange(0, d - 1)
        b = rng.randrange(a + 1, d)
        if (a, b) not in topo_edges and indeg[b] < max_parents:
            topo_edges.append((a, b)); indeg[b] += 1
        tries += 1

    # 2) leak + noisy-OR weights
    leak = [rng.uniform(leak_lo, leak_hi) for _ in range(d)]
    w = {(a, b): rng.uniform(w_lo, w_hi) for (a, b) in topo_edges}
    parents = {b: [a for (a, bb) in topo_edges if bb == b] for b in range(d)}

    # 3) draw tours in topological order
    Xtopo = [[0] * d for _ in range(n)]
    for b in range(d):
        pl = parents.get(b, [])
        lb = leak[b]
        for r in range(n):
            keep = 1.0 - lb
            row = Xtopo[r]
            for a in pl:
                if row[a] == 1:
                    keep *= (1.0 - w[(a, b)])
            p1 = 1.0 - keep
            row[b] = 1 if rng.random() < p1 else 0

    # 4) relabel topo index -> data column via a random permutation
    perm = list(range(d))
    rng.shuffle(perm)                 # perm[topo_index] = data_column
    data = [[0] * d for _ in range(n)]
    for r in range(n):
        srow = Xtopo[r]
        drow = data[r]
        for t in range(d):
            drow[perm[t]] = srow[t]
    true_edges = set((perm[a], perm[b]) for (a, b) in topo_edges)
    return true_edges, data


def _build_instances():
    specs = [
        dict(seed=39801, d=6, n=420, edge_prob=0.55),
        dict(seed=39802, d=6, n=320, edge_prob=0.50),   # tour-poorer
        dict(seed=39803, d=7, n=440, edge_prob=0.52),
        dict(seed=39824, d=7, n=300, edge_prob=0.48),   # tour-poorer
        dict(seed=39827, d=8, n=460, edge_prob=0.48),
        dict(seed=39806, d=8, n=320, edge_prob=0.44),   # tour-poorer
        dict(seed=39807, d=9, n=480, edge_prob=0.42),   # held-out: larger
        dict(seed=39808, d=9, n=320, edge_prob=0.40),   # held-out: larger + tour-poorer
        dict(seed=39809, d=10, n=500, edge_prob=0.40),  # held-out: largest
        dict(seed=39810, d=10, n=340, edge_prob=0.38),  # held-out: largest + tour-poorer
    ]
    out = []
    for p in specs:
        true_edges, data = _gen_bn(**p)
        out.append({"name": f"mus{p['seed']}", "true_edges": true_edges,
                    "data": data, "d": p["d"], "n": p["n"]})
    return out


# ============================ structural Hamming distance ==================
def _shd(true_edges, pred_edges, d):
    """SHD over unordered pairs: missing / extra / reversed-or-ambiguous each = 1."""
    shd = 0
    for i in range(d):
        for j in range(i + 1, d):
            t_ij = (i, j) in true_edges
            t_ji = (j, i) in true_edges
            p_ij = (i, j) in pred_edges
            p_ji = (j, i) in pred_edges
            t_any = t_ij or t_ji
            p_any = p_ij or p_ji
            if not t_any and not p_any:
                continue
            if t_any != p_any:                # skeleton mismatch (missing or extra)
                shd += 1
                continue
            if t_ij and p_ij and not t_ji and not p_ji:
                continue
            if t_ji and p_ji and not t_ij and not p_ij:
                continue
            shd += 1                          # reversed / ambiguous orientation
    return shd


# ============================ candidate answer handling ====================
def _parse_pred(ans, d):
    """Return a set of directed edges (i,j) or None if the answer is malformed."""
    if isinstance(ans, dict):
        if "edges" in ans:
            ans = ans["edges"]
        elif "adjacency" in ans:
            ans = ans["adjacency"]
        else:
            return None
    if not isinstance(ans, list):
        return None

    # adjacency-matrix form: exactly d rows, each a length-d list of 0/1
    if len(ans) == d and all(isinstance(r, list) and len(r) == d for r in ans):
        edges = set()
        for i in range(d):
            for j in range(d):
                v = ans[i][j]
                if isinstance(v, bool):
                    v = 1.0 if v else 0.0
                elif isinstance(v, (int, float)):
                    v = float(v)
                else:
                    return None
                if v not in (0.0, 1.0):
                    return None
                if v == 1.0:
                    if i == j:
                        return None
                    edges.add((i, j))
        return edges

    # edge-list form: each entry is a [i, j] pair of in-range distinct ints
    edges = set()
    for e in ans:
        if not (isinstance(e, list) and len(e) == 2):
            return None
        a, b = e
        if isinstance(a, bool) or isinstance(b, bool):
            return None
        if not (isinstance(a, int) and isinstance(b, int)):
            return None
        if not (0 <= a < d) or not (0 <= b < d) or a == b:
            return None
        edges.add((a, b))
    return edges


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        d = inst["d"]
        true_edges = inst["true_edges"]
        E = len(true_edges)                       # SHD of the empty-graph baseline
        data = inst["data"]

        public = {
            "data": data,
            "n_tours": int(inst["n"]),
            "n_galleries": int(d),
            "gallery_names": [_GALLERIES[k % len(_GALLERIES)] for k in range(d)],
            "seed": int(20240398 + inst["n"] + d),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue

        pred = _parse_pred(ans, d)
        if pred is None:
            vec.append(0.0)
            continue

        try:
            shd = _shd(true_edges, pred, d)
        except Exception:
            vec.append(0.0)
            continue

        r = min(1.0, 0.1 * float(E) / max(float(shd), 1e-9))
        if not (r == r) or r < 0.0:
            r = 0.0
        if 0.0 < r < VALID_FLOOR:
            r = VALID_FLOOR
        vec.append(float(r))

    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
