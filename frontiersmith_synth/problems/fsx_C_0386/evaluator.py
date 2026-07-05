#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0386 -- "Asteroid Rig: Causal Map of Subsystem Telemetry"
(family: causal-discovery; eval_form: quality-metric; MLS-Bench causal-* shape).

An autonomous asteroid-mining rig streams DISCRETE status codes from a bank of
subsystems every duty cycle -- drill speed, regolith flow, ore grade, dust
density, reactor temperature, coolant pressure, thruster load, comms latency,
power draw, gyro drift, hopper fill, solar flux.  Each subsystem reports a small
categorical code (0 = low / 1 = nominal / 2 = high).  These codes are NOT
independent: they influence one another through a HIDDEN causal wiring diagram --
a directed acyclic graph (DAG) in which each subsystem's code is a discretized,
noisy function of its direct-cause subsystems.  Only OBSERVATIONAL telemetry is
available (you cannot command the drill or heat the reactor to watch the effect).

The candidate designs a CAUSAL-DISCOVERY routine over categorical data.  It is run
as an ISOLATED subprocess (isorun): it reads ONE JSON "public instance" from stdin
(the raw N x d integer telemetry table + node count + per-node cardinality) and
writes ONE JSON answer (a predicted set of directed edges) to stdout.  It NEVER
sees the ground-truth DAG, the topological order, the structural weights, the
latent variables, or this evaluator's memory.

Public instance JSON (stdin):
    {
      "data":         [[int, ...], ...],   # N x d integer telemetry (fresh copy), each entry in 0..K-1
      "n_samples":    int,                 # number of duty cycles N
      "n_nodes":      int,                 # number of subsystems d
      "n_categories": int,                 # global category count K (codes are 0..K-1)
      "cardinalities":[int, ...],          # per-column category count (max code + 1), a convenience
      "node_names":   [str, ...],          # flavor labels for the d subsystems
      "seed":         int                  # per-instance seed the candidate MAY use for its own RNG
    }

Answer JSON (stdout) -- ANY of these equivalent forms is accepted:
    [[i, j], ...]                          # directed edges  i -> j  (subsystem i causes subsystem j)
    {"edges": [[i, j], ...]}               # same, wrapped
    {"adjacency": [[0/1, ...], ...]}       # d x d adjacency matrix, A[i][j]=1 means i -> j

Quality is measured by the STRUCTURAL HAMMING DISTANCE (SHD) between the predicted
graph and the hidden ground-truth DAG, recomputed deterministically here.  SHD
counts, over every unordered subsystem pair, the edits needed to match truth: a
MISSING edge, an EXTRA edge, or a REVERSED / ambiguous orientation each cost 1.
Lower SHD is better.

The evaluator's own trivial baseline is the EMPTY graph, whose SHD equals the
number of true edges E.  Per instance the normalized score is the minimization
form from the authoring contract:

    r = min( 1.0, 0.1 * E / max(SHD_cand, 1e-9) )

so a candidate that predicts no edges (or is otherwise no better than empty) maps
to ~0.1, and a perfect recovery (SHD = 0) maps to 1.0.  Because only OBSERVATIONAL
categorical data is given, exact orientation is under-determined, indirect
(transitive) associations masquerade as direct edges unless screened off by
conditional-independence tests, and finite samples blur the weakest links -- so
even a strong routine leaves headroom below 1.0 on the larger, data-poorer,
denser rigs.  The final score is the MEAN of the per-instance r over a diverse
battery of rigs (sparse/dense, few/many subsystems, data-rich/data-poor, plus
larger held-out rigs), rewarding a discovery rule that GENERALIZES.

Valid instances are floored to a small positive value; an instance where the
candidate crashes, times out, or returns a malformed / out-of-range answer scores
exactly 0.0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import os
# Cap BLAS/OMP threads so numpy imports cleanly inside isorun's memory-capped child.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys, json
import numpy as np
import isorun

VALID_FLOOR = 0.02      # floor for VALID instances (so a weak-but-valid answer stays > invalid=0)
CAND_TIMEOUT = 20
K_GLOBAL = 3            # category count: 0 = low, 1 = nominal, 2 = high
_THRESH = [-0.9, 0.9]  # FIXED discretization thresholds shared across all subsystems

_SUBSYSTEMS = [
    "drill_rpm", "regolith_flow", "ore_grade", "dust_density", "reactor_temp",
    "coolant_pressure", "thruster_load", "comms_latency", "power_draw",
    "gyro_drift", "hopper_fill", "solar_flux",
]


# ============================ asteroid-rig family (instances) ==============
def _gen_discrete_bn(seed, d, n, edge_prob, w_lo=0.6, w_hi=1.0):
    """Generate a discrete causal Bayesian-network rig.

    A latent linear-Gaussian SEM is drawn over a random DAG on d subsystems with
    EQUAL-variance disturbances, then each subsystem's latent is discretized into
    K=3 codes with the SAME fixed thresholds (_THRESH).  Because the thresholds
    are shared, a shallow (source) subsystem -- unit-variance latent -- lands
    mostly in the central 'nominal' bin (lower marginal entropy), while a deep
    (descendant) subsystem -- larger latent variance from accumulated causes --
    spreads across the outer 'low'/'high' bins (higher marginal entropy).  That
    ascending-entropy gradient is a genuine (but noisy) topological-order signal.

    Direct causal links give strong pairwise association; indirect (transitive)
    paths give weaker association that a marginal test mistakes for a real edge
    unless screened off by conditioning; co-parents of a collider are marginally
    independent, so a marginal-then-condition skeleton stays clean there.  The
    DAG is built in topological order then RELABELLED by a random permutation, so
    the column index carries NO orientation information.

    Returns (true_edges, data) where true_edges is a set of (i,j) in DATA-column
    space meaning subsystem i -> subsystem j, and data is an N x d int array with
    entries in 0..K-1.
    """
    rng = np.random.default_rng(seed)
    # 1) edges in topological space (topo node a -> topo node b, a < b)
    topo_edges = []
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < edge_prob:
                topo_edges.append((a, b))
    tries = 0
    while len(topo_edges) < 2 and tries < 50:
        a = int(rng.integers(0, d - 1))
        b = int(rng.integers(a + 1, d))
        if (a, b) not in topo_edges:
            topo_edges.append((a, b))
        tries += 1

    # 2) structural weights (bounded away from 0, random sign)
    W = np.zeros((d, d))
    for (a, b) in topo_edges:
        mag = rng.uniform(w_lo, w_hi)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        W[a, b] = sign * mag

    # 3) latent linear-Gaussian draw in topological order, equal-variance noise
    Z = np.zeros((n, d))
    for b in range(d):
        contrib = np.zeros(n)
        for a in range(b):
            if W[a, b] != 0.0:
                contrib = contrib + W[a, b] * Z[:, a]
        Z[:, b] = contrib + rng.normal(0.0, 1.0, size=n)

    # 4) discretize with the SHARED fixed thresholds -> codes in 0..K-1
    codes = np.digitize(Z, _THRESH)          # 0,1,2 for K=3

    # 5) relabel topo index -> data column via a random permutation
    perm = rng.permutation(d)
    data = np.zeros((n, d), dtype=np.int64)
    for t in range(d):
        data[:, perm[t]] = codes[:, t]
    true_edges = set()
    for (a, b) in topo_edges:
        true_edges.add((int(perm[a]), int(perm[b])))
    return true_edges, data


def _build_instances():
    specs = [
        dict(seed=311, d=6, n=450, edge_prob=0.34),
        dict(seed=312, d=6, n=380, edge_prob=0.30),
        dict(seed=313, d=7, n=340, edge_prob=0.40),   # denser + data-poorer
        dict(seed=314, d=7, n=360, edge_prob=0.32),
        dict(seed=315, d=8, n=360, edge_prob=0.36),   # denser + data-poorer
        dict(seed=316, d=8, n=420, edge_prob=0.36),   # denser
        dict(seed=317, d=9, n=700, edge_prob=0.26),   # held-out: larger
        dict(seed=318, d=9, n=440, edge_prob=0.28),   # held-out: larger + data-poorer
        dict(seed=319, d=10, n=760, edge_prob=0.24),  # held-out: largest
        dict(seed=320, d=10, n=480, edge_prob=0.26),  # held-out: largest + data-poorer
    ]
    out = []
    for p in specs:
        true_edges, data = _gen_discrete_bn(**p)
        out.append({"name": f"rig{p['seed']}", "true_edges": true_edges,
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
            if t_any != p_any:            # one has a skeleton edge, the other doesn't
                shd += 1
                continue
            # both have a skeleton edge here: match only if same single orientation
            if t_ij and p_ij and not t_ji and not p_ji:
                continue
            if t_ji and p_ji and not t_ij and not p_ij:
                continue
            shd += 1
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
                if v != v or v not in (0.0, 1.0):
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
        cardinalities = [int(data[:, c].max()) + 1 for c in range(d)]

        public = {
            "data": data.tolist(),
            "n_samples": int(inst["n"]),
            "n_nodes": int(d),
            "n_categories": int(K_GLOBAL),
            "cardinalities": cardinalities,
            "node_names": [_SUBSYSTEMS[k % len(_SUBSYSTEMS)] for k in range(d)],
            "seed": int(20240386 + inst["n"] + d),
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
