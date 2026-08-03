#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0182 -- "Telescope Array: Causal Wiring of Sensor Channels"
(family: causal-discovery; eval_form: quality-metric; MLS-Bench causal-* shape).

A robotic telescope array logs a bank of environmental / control channels every
exposure (dome seeing, ambient & mirror temperature, wind, humidity, mount
tracking error, guide-star drift, focus position, CCD temperature, sky
background, ...).  These channels influence one another through a HIDDEN causal
wiring diagram: a directed acyclic graph (DAG) in which each channel is a linear
function of its direct causes plus independent Gaussian disturbance (a
linear-Gaussian structural equation model).  Only OBSERVATIONAL logs are
available -- no interventions.

The candidate designs a CAUSAL-DISCOVERY routine.  It is run as an ISOLATED
subprocess (isorun): it reads ONE JSON "public instance" from stdin (the raw
observation matrix + node count) and writes ONE JSON answer (a predicted set of
directed edges) to stdout.  It NEVER sees the ground-truth DAG, the topological
order, the edge weights, or this evaluator's memory.

Public instance JSON (stdin):
    {
      "data":       [[float, ...], ...],   # N x d observation matrix (fresh copy)
      "n_samples":  int,                   # number of exposures N
      "n_nodes":    int,                   # number of channels d
      "node_names": [str, ...],            # flavor labels for the d channels
      "seed":       int                    # per-instance seed the candidate MAY use
    }

Answer JSON (stdout) -- EITHER form is accepted:
    [[i, j], ...]                          # directed edges  i -> j  (channel i causes channel j)
    {"edges": [[i, j], ...]}               # same, wrapped
    {"adjacency": [[0/1, ...], ...]}       # d x d adjacency matrix, A[i][j]=1 means i -> j

Quality is measured by the STRUCTURAL HAMMING DISTANCE (SHD) between the predicted
graph and the hidden ground-truth DAG, recomputed deterministically here.  SHD
counts, over every unordered channel pair, the edits needed to match truth:
a MISSING edge, an EXTRA edge, or a REVERSED / ambiguous orientation each cost 1.
Lower SHD is better.

The evaluator's own trivial baseline is the EMPTY graph, whose SHD equals the
number of true edges E.  Per instance the normalized score is the minimization
form from the authoring contract:

    r = min( 1.0, 0.1 * SHD_empty / max(SHD_cand, 1e-9) )

so a candidate that predicts no edges (or is otherwise no better than empty) maps
to ~0.1, and a perfect recovery (SHD = 0) maps to 1.0.  Because only OBSERVATIONAL
data is given, exact orientation is generally under-determined and a
precision-matrix skeleton also picks up "moralization" edges between co-parents,
so even a strong routine leaves headroom below 1.0 on the harder arrays.  The
final score is the MEAN of the per-instance r over a diverse battery of arrays
(sparse/dense, few/many channels, data-rich/data-poor, plus held-out larger
arrays), rewarding a discovery rule that GENERALIZES.

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

_CHANNELS = [
    "dome_seeing", "ambient_temp", "mirror_temp", "wind_speed", "humidity",
    "mount_track_err", "guide_drift", "focus_pos", "ccd_temp", "sky_bkg",
    "dew_point", "airmass",
]


# ============================ telescope-array family (instances) ============
def _gen_dag_sem(seed, d, n, edge_prob, w_lo=0.6, w_hi=1.3):
    """Generate a random linear-Gaussian SEM over a DAG on d channels and draw N
    observations.  The DAG is built in a topological order then RELABELLED by a
    random permutation, so plain index order carries NO orientation information.
    Equal-variance disturbances make the ordering weakly recoverable (a source
    channel has smaller marginal variance than its descendants), which is exactly
    the sort of structure a good routine should exploit -- but colliders create
    moralization edges that keep the problem from being trivial.
    Returns (true_edges, data) where true_edges is a set of (i,j) in DATA-column
    space meaning channel i -> channel j, and data is an N x d float64 array.
    """
    rng = np.random.default_rng(seed)
    # 1) build edges in topological space (topo node a -> topo node b, a < b)
    topo_edges = []
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < edge_prob:
                topo_edges.append((a, b))
    # guarantee at least 2 edges so the empty-graph baseline is well defined
    tries = 0
    while len(topo_edges) < 2 and tries < 50:
        a = int(rng.integers(0, d - 1))
        b = int(rng.integers(a + 1, d))
        if (a, b) not in topo_edges:
            topo_edges.append((a, b))
        tries += 1

    # 2) edge weights (bounded away from 0, random sign)
    W = np.zeros((d, d))          # W[a,b] weight of topo edge a -> b
    for (a, b) in topo_edges:
        mag = rng.uniform(w_lo, w_hi)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        W[a, b] = sign * mag

    # 3) draw data in topological order, equal-variance Gaussian disturbances
    Xtopo = np.zeros((n, d))
    for b in range(d):
        parents = [a for a in range(b) if W[a, b] != 0.0]
        contrib = np.zeros(n)
        for a in parents:
            contrib = contrib + W[a, b] * Xtopo[:, a]
        Xtopo[:, b] = contrib + rng.normal(0.0, 1.0, size=n)

    # 4) relabel topo index -> data column via a random permutation
    perm = rng.permutation(d)     # perm[topo_index] = data_column
    data = np.zeros((n, d))
    for t in range(d):
        data[:, perm[t]] = Xtopo[:, t]
    true_edges = set()
    for (a, b) in topo_edges:
        true_edges.add((int(perm[a]), int(perm[b])))
    return true_edges, data.astype(np.float64)


def _build_instances():
    specs = [
        dict(seed=311, d=6, n=500, edge_prob=0.35),
        dict(seed=312, d=6, n=400, edge_prob=0.30),
        dict(seed=313, d=7, n=320, edge_prob=0.42),   # denser (more colliders) + data-poorer
        dict(seed=314, d=7, n=350, edge_prob=0.35),   # data-poorer
        dict(seed=315, d=8, n=340, edge_prob=0.40),   # denser (more colliders) + data-poorer
        dict(seed=316, d=8, n=400, edge_prob=0.32),   # denser + data-poorer
        dict(seed=317, d=9, n=800, edge_prob=0.25),   # held-out: larger
        dict(seed=318, d=9, n=450, edge_prob=0.30),   # held-out: larger + data-poorer
        dict(seed=319, d=10, n=800, edge_prob=0.24),  # held-out: largest
        dict(seed=320, d=10, n=500, edge_prob=0.28),  # held-out: largest + data-poorer
    ]
    out = []
    for p in specs:
        true_edges, data = _gen_dag_sem(**p)
        out.append({"name": f"arr{p['seed']}", "true_edges": true_edges,
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
            "data": data.tolist(),
            "n_samples": int(inst["n"]),
            "n_nodes": int(d),
            "node_names": [_CHANNELS[k % len(_CHANNELS)] for k in range(d)],
            "seed": int(20240182 + inst["n"] + d),
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
