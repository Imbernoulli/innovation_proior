#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0392 -- "Cold-Chain Causal Cartographer"
(family: causal-discovery; format B, quality-metric; theme: vaccine cold chain).

THEME.  A national vaccine cold-chain programme instruments every fridge, freezer,
transport box and courier hand-off with sensors and audit fields.  Each shipment
produces one record of DISCRETE status variables -- e.g. ambient temperature band,
compressor duty state, door-open events, coolant-pack charge, GPS route class, a
customs-hold flag, sensor-fault indicators, and a final "vaccine viable?" verdict.
The variables are causally linked (a failing compressor RAISES the temperature band,
which TRIPS an excursion alarm and DEGRADES viability), but the programme only ever
logs OBSERVATIONAL snapshots -- never a controlled experiment.

You are handed a large batch of such records for ONE deployment region and must
RECONSTRUCT the causal wiring diagram: which status variable directly drives which.
Formally, recover the directed acyclic graph (DAG) of the discrete Bayesian network
that generated the data.  The truth is hidden; you see only the sampled records.

This is a discrete causal-discovery / structure-learning task (MLS-Bench `causal-*`
shape) scored by the classic Structural Hamming Distance (SHD) to the ground-truth
graph, aggregated over a battery of independently-generated regions of growing size.
The multi-region aggregate forces a GENERALIZABLE discovery rule -- constraint-based
(conditional-independence / PC-style) methods, score-based search, and pure
association thresholding all give genuinely different SHD, and none reach 0 at the
sample sizes provided, so there is real headroom.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n": n,                         # number of status variables
             "card": [c_0, ..., c_{n-1}],    # category count of each variable (2 or 3)
             "data": [[x_0, ..., x_{n-1}], ...]}   # m records, each n category ints
          NOTE: variable indices are RANDOMLY relabelled -- index order carries NO
          information about causal (topological) order.  The ground-truth graph and
          all record generation live only in THIS parent process; the candidate
          never sees them.
  stdout: ONE JSON object:
            {"edges": [[i, j], ...]}         # estimated DIRECTED edges  i -> j

  An estimate is VALID iff "edges" is a list of [i,j] integer pairs with
  0 <= i,j < n, i != j, NO duplicate directed edge, and NEVER both i->j and j->i
  for the same unordered pair.  A malformed estimate, a self-loop, a duplicated or
  bidirected pair, an out-of-range index, a crash, a timeout, or non-JSON -> that
  region scores 0.0.  The empty estimate {"edges": []} is VALID (it recovers no
  wiring; it scores the calibrated ~0.1 baseline).

SCORING (deterministic; no wall-time).  Per region the evaluator computes the
Structural Hamming Distance between the candidate's directed edge set E_hat and the
true edge set E_true, over all unordered variable pairs {i,j}:
    * true edge present, estimate absent            -> +1  (a MISSED link)
    * true edge absent,  estimate present           -> +1  (a SPURIOUS link)
    * both present but in OPPOSITE directions       -> +1  (a REVERSED link)
    * both present, same direction, or both absent  ->  0
    SHD = sum of the above.
The weak reference is the EMPTY graph, whose SHD equals the number of true edges E:
    base = |E_true|                                   # SHD of predicting nothing
and the score normalises a minimisation objective with an affine anchor:
    r = clamp( 0.1 * base / max(SHD, 1e-12), 0, 1 )
so predicting nothing scores ~0.1, halving the SHD scores ~0.2, driving SHD toward 0
approaches 1.0, and a worse-than-nothing estimate (SHD > E) scores < 0.1.  Because
the records are only observational and finite, orientation is genuinely ambiguous
(Markov-equivalence) and skeleton recovery is imperfect -> even strong discovery
stays well below 1.0, leaving headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The ground-truth
DAG, the CPTs, and the SHD scorer live only in THIS parent process, so a
frame-walking / filesystem-snooping candidate learns nothing about the answer.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all regions, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return state

    def nxt_int(lo, hi):                       # inclusive
        return lo + (nxt() >> 17) % (hi - lo + 1)

    def nxt_float():                           # in [0,1)
        return (nxt() >> 11) / float(1 << 53)

    return nxt_int, nxt_float


# ----------------------------- instance family -----------------------------
def _build_region(seed, n, dens, m, sharp, maxpar):
    """Build one discrete Bayesian network over `n` variables in internal topo order
    0..n-1 (parents always have smaller index), forward-sample `m` records, then
    RANDOMLY RELABEL the variables so public index order leaks no causal order.

    Returns (public_dict, true_edges_public)."""
    ni, nf = _rng(seed)

    # cardinalities (2 or 3 categories each)
    card = [2 if nf() < 0.6 else 3 for _ in range(n)]

    # parent sets in topo order
    parents = [[] for _ in range(n)]
    edges_topo = []
    for j in range(1, n):
        cand = list(range(j))
        picked = [i for i in cand if nf() < dens]
        # cap in-degree so CPTs stay small; keep the highest-index (nearest) parents
        if len(picked) > maxpar:
            picked = sorted(picked)[-maxpar:]
        parents[j] = picked
        for i in picked:
            edges_topo.append((i, j))
    # guarantee at least one edge
    if not edges_topo:
        parents[1] = [0]
        edges_topo.append((0, 1))

    # conditional prob tables: for each node, for each parent-config a peaky categorical
    def peaky(c):
        w = [nf() ** sharp for _ in range(c)]
        s = sum(w)
        if s <= 0:
            return [1.0 / c] * c
        return [x / s for x in w]

    cpt = []
    for node in range(n):
        pc = [card[p] for p in parents[node]]
        nconf = 1
        for x in pc:
            nconf *= x
        table = [peaky(card[node]) for _ in range(nconf)]
        cpt.append(table)

    def conf_index(node, pv):
        # mixed-radix index of the parent value tuple pv over card[parents]
        idx = 0
        for p, v in zip(parents[node], pv):
            idx = idx * card[p] + v
        return idx

    def sample_cat(dist):
        u = nf()
        acc = 0.0
        for k, pk in enumerate(dist):
            acc += pk
            if u < acc:
                return k
        return len(dist) - 1

    # forward sample m records in topo order
    data_topo = []
    for _ in range(m):
        row = [0] * n
        for node in range(n):
            pv = [row[p] for p in parents[node]]
            dist = cpt[node][conf_index(node, pv)]
            row[node] = sample_cat(dist)
        data_topo.append(row)

    # random relabelling: perm[old] = new index
    perm = list(range(n))
    for a in range(n - 1, 0, -1):
        b = ni(0, a)
        perm[a], perm[b] = perm[b], perm[a]

    card_pub = [0] * n
    for old in range(n):
        card_pub[perm[old]] = card[old]

    data_pub = []
    for row in data_topo:
        nr = [0] * n
        for old in range(n):
            nr[perm[old]] = row[old]
        data_pub.append(nr)

    true_edges = [[perm[i], perm[j]] for (i, j) in edges_topo]

    public = {"name": f"region{seed}", "n": n, "card": card_pub, "data": data_pub}
    return public, true_edges


def _build_instances():
    """Deterministic battery of regions (seed, n, dens, m, sharp, maxpar)."""
    specs = [
        (31, 10, 0.26, 1300, 3.4, 2),
        (32, 12, 0.24, 1300, 3.4, 2),
        (33, 12, 0.28, 1200, 3.2, 2),
        (34, 14, 0.22, 1500, 3.4, 2),
        (35, 14, 0.26, 1200, 3.2, 2),
        (36, 16, 0.20, 1500, 3.3, 2),
        (37, 16, 0.24, 1300, 3.2, 2),
        (38, 18, 0.20, 1500, 3.3, 2),
        # harder / larger held-out regions (denser and/or fewer records)
        (41, 16, 0.28, 1000, 3.0, 2),
        (42, 18, 0.24, 1100, 3.0, 2),
        (43, 20, 0.20, 1200, 3.1, 2),
        (44, 20, 0.24, 1100, 3.0, 2),
    ]
    out = []
    for (seed, n, dens, m, sharp, maxpar) in specs:
        public, true_edges = _build_region(seed, n, dens, m, sharp, maxpar)
        out.append({"public": public, "true_edges": true_edges})
    return out


# ----------------------------- validation / scoring ------------------------
def _est_edge_set(answer, n):
    """Validate the candidate estimate. Return a set of directed (i,j) tuples, or
    None if the estimate is malformed / infeasible."""
    if not isinstance(answer, dict):
        return None
    edges = answer.get("edges")
    if not isinstance(edges, list):
        return None
    directed = set()
    pairs = set()
    for item in edges:
        if not isinstance(item, list) or len(item) != 2:
            return None
        i, j = item
        if isinstance(i, bool) or isinstance(j, bool):
            return None
        if not isinstance(i, int) or not isinstance(j, int):
            return None
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return None
        if (i, j) in directed:
            return None                        # duplicate directed edge
        key = (i, j) if i < j else (j, i)
        if key in pairs:
            return None                        # both i->j and j->i for one pair
        pairs.add(key)
        directed.add((i, j))
    return directed


def _shd(true_edges, est):
    """Structural Hamming Distance over unordered pairs."""
    true_set = set((i, j) for (i, j) in true_edges)
    shd = 0
    seen_pairs = set()
    # pairs touched by either graph
    all_pairs = set()
    for (i, j) in true_set:
        all_pairs.add((i, j) if i < j else (j, i))
    for (i, j) in est:
        all_pairs.add((i, j) if i < j else (j, i))
    for (a, b) in all_pairs:                    # a < b
        t = 1 if (a, b) in true_set else (2 if (b, a) in true_set else 0)
        e = 1 if (a, b) in est else (2 if (b, a) in est else 0)
        if t == e:
            continue
        shd += 1                               # missing, spurious, or reversed
    return shd


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = inst["public"]
        true_edges = inst["true_edges"]
        base = len(true_edges)                 # SHD of the empty graph
        if base < 1:
            base = 1
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            est = _est_edge_set(ans, public["n"])
        except Exception:
            est = None
        if est is None:
            vec.append(0.0)
            continue
        shd = _shd(true_edges, est)
        r = 0.1 * base / max(shd, 1e-12)
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
