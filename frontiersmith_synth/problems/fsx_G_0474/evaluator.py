#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0474 -- "Riverside Sensor Mesh: Recovering the Causal
Wiring of an Environmental Monitoring Network"
(family: ml-causal-discovery; eval_form: quality-metric; MLS-Bench causal-* shape).

A solar-powered WIRELESS SENSOR MESH is deployed along a river reach.  Each node
in the mesh streams an environmental channel -- rainfall, incident solar
radiation, air temperature, relative humidity, soil moisture, leaf wetness,
canopy temperature, stream flow, turbidity, water pH, dissolved oxygen, wind
speed, ... -- once per logging interval.  These channels influence one another
through a HIDDEN causal wiring diagram: a directed acyclic graph (DAG) in which
each channel is a linear function of its direct causes plus an INDEPENDENT
Gaussian disturbance whose scale VARIES from channel to channel (a
heteroscedastic linear-Gaussian structural equation model).  Only passive
OBSERVATIONAL logs are available; no interventions were performed.

A distinctive feature of the mesh: a few channels are known WEATHER-FORCING
inputs (e.g. rainfall, solar radiation, wind) -- physical drivers that have NO
upstream sensor cause.  The identity (column indices) of these forcing channels
is published to the solver as a partial-orientation constraint: any edge touching
a forcing channel must point AWAY from it.  A good routine exploits this anchor.

The candidate designs a CAUSAL-DISCOVERY routine.  It is run as an ISOLATED
subprocess (isorun): it reads ONE JSON "public instance" from stdin (the raw
observation matrix, the channel count, and the forcing-node indices) and writes
ONE JSON answer (a predicted set of DIRECTED edges) to stdout.  It NEVER sees the
ground-truth DAG, the topological order, the disturbance scales, the edge
weights, or this evaluator's memory.

Public instance JSON (stdin):
    {
      "data":          [[float, ...], ...],  # N x d observation matrix (fresh copy)
      "n_samples":     int,                  # number of logging intervals N
      "n_nodes":       int,                  # number of channels d
      "node_names":    [str, ...],           # flavor labels for the d channels
      "forcing_nodes": [int, ...],           # channels known to have NO parents (weather drivers)
      "seed":          int                   # per-instance seed the candidate MAY use
    }

Answer JSON (stdout) -- ANY of these forms is accepted:
    [[i, j], ...]                            # directed edges  i -> j  (channel i causes channel j)
    {"edges": [[i, j], ...]}                 # same, wrapped
    {"adjacency": [[0/1, ...], ...]}         # d x d adjacency matrix, A[i][j]=1 means i -> j

Quality is the STRUCTURAL HAMMING DISTANCE (SHD) between the predicted graph and
the hidden ground-truth DAG, recomputed deterministically here.  SHD counts, over
every unordered channel pair, the edits needed to match truth: a MISSING edge, an
EXTRA edge, or a REVERSED / ambiguous orientation each cost 1.  Lower SHD is
better (objective = minimize).

The evaluator's own trivial baseline is the EMPTY graph, whose SHD equals the
number of true edges E.  Per instance the normalized score is the minimization
form from the authoring contract:

    r = min( 1.0, 0.1 * SHD_empty / max(SHD_cand, 1e-9) )

so a candidate no better than the empty graph maps to ~0.1 and a perfect recovery
(SHD = 0) maps to 1.0.  Because only OBSERVATIONAL data is given, orientation is
generally under-determined, the heteroscedastic disturbances defeat the naive
"a source has the smallest marginal variance" trick, and a precision-matrix
skeleton also picks up "moralization" edges between co-parents -- so even a strong
routine leaves headroom below 1.0 on the harder meshes.  The final score is the
MEAN of the per-instance r over a diverse battery of meshes (sparse/dense,
few/many channels, data-rich/data-poor, plus held-out larger meshes), rewarding a
discovery rule that GENERALIZES.

An instance where the candidate crashes, times out, or returns a malformed /
out-of-range answer scores exactly 0.0; a valid but weak answer is floored to a
small positive value so it stays strictly above the crash floor.

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

VALID_FLOOR = 0.02      # floor for VALID instances (weak-but-valid stays > invalid=0)
CAND_TIMEOUT = 20

_CHANNELS = [
    "rainfall", "solar_rad", "air_temp", "rel_humidity", "soil_moisture",
    "leaf_wetness", "canopy_temp", "stream_flow", "turbidity", "water_ph",
    "dissolved_o2", "wind_speed",
]


# ===================== riverside sensor-mesh family (instances) =============
def _gen_dag_sem(seed, d, n, edge_prob, n_exo, w_lo=0.6, w_hi=1.4,
                 s_lo=0.5, s_hi=2.0):
    """Random heteroscedastic linear-Gaussian SEM over a DAG on d channels.

    The DAG is built in a topological order in which the FIRST `n_exo` topo nodes
    are declared exogenous WEATHER FORCINGS: they receive no parents but each gets
    at least one child, so they are genuine informative roots.  Every non-forcing
    node draws parents from strictly-earlier topo nodes.  Each node has its OWN
    disturbance scale drawn from U(s_lo, s_hi) (heteroscedastic), which is what
    defeats the "smallest-marginal-variance = source" heuristic.  The topo order
    is then RELABELLED by a random permutation, so plain column order carries NO
    orientation information.

    Returns (true_edges, forcing_cols, data):
      true_edges  : set of (i,j) in DATA-column space, meaning channel i -> j
      forcing_cols: sorted list of data columns that are exogenous forcings
      data        : N x d float64 observation matrix
    """
    rng = np.random.default_rng(seed)
    n_exo = max(1, min(n_exo, d - 1))

    # 1) edges in topological space (a -> b with a < b); forcings (a < n_exo) get
    #    NO incoming edges by construction (b must be >= n_exo to receive a parent).
    topo_edges = []
    for b in range(n_exo, d):
        for a in range(b):
            if rng.random() < edge_prob:
                topo_edges.append((a, b))
    # ensure every forcing node has >= 1 child so it is informative
    child_of = {a: [b for (aa, b) in topo_edges if aa == a] for a in range(n_exo)}
    for a in range(n_exo):
        if not child_of.get(a):
            b = int(rng.integers(max(a + 1, n_exo), d))
            topo_edges.append((a, b))
    # guarantee at least 2 edges so the empty-graph baseline is well defined
    tries = 0
    while len(topo_edges) < 2 and tries < 50:
        b = int(rng.integers(n_exo, d))
        a = int(rng.integers(0, b))
        if (a, b) not in topo_edges:
            topo_edges.append((a, b))
        tries += 1
    topo_edges = sorted(set(topo_edges))

    # 2) edge weights (bounded away from 0, random sign)
    W = np.zeros((d, d))
    for (a, b) in topo_edges:
        mag = rng.uniform(w_lo, w_hi)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        W[a, b] = sign * mag

    # 3) per-node disturbance scales (HETEROSCEDASTIC)
    sigma = rng.uniform(s_lo, s_hi, size=d)

    # 4) draw data in topological order
    Xtopo = np.zeros((n, d))
    for b in range(d):
        contrib = np.zeros(n)
        for a in range(b):
            if W[a, b] != 0.0:
                contrib = contrib + W[a, b] * Xtopo[:, a]
        Xtopo[:, b] = contrib + rng.normal(0.0, sigma[b], size=n)

    # 5) relabel topo index -> data column via a random permutation
    perm = rng.permutation(d)            # perm[topo_index] = data_column
    data = np.zeros((n, d))
    for t in range(d):
        data[:, perm[t]] = Xtopo[:, t]
    true_edges = set((int(perm[a]), int(perm[b])) for (a, b) in topo_edges)
    forcing_cols = sorted(int(perm[t]) for t in range(n_exo))
    return true_edges, forcing_cols, data.astype(np.float64)


def _build_instances():
    specs = [
        dict(seed=4741, d=6, n=500, edge_prob=0.35, n_exo=2),
        dict(seed=4742, d=6, n=380, edge_prob=0.32, n_exo=2),   # data-poorer
        dict(seed=4743, d=7, n=320, edge_prob=0.40, n_exo=2),   # denser + data-poorer
        dict(seed=4744, d=7, n=420, edge_prob=0.34, n_exo=3),
        dict(seed=4745, d=8, n=340, edge_prob=0.38, n_exo=2),   # denser + data-poorer
        dict(seed=4746, d=8, n=460, edge_prob=0.30, n_exo=3),
        dict(seed=4747, d=9, n=800, edge_prob=0.26, n_exo=3),   # held-out: larger
        dict(seed=4748, d=9, n=440, edge_prob=0.30, n_exo=2),   # held-out: larger + data-poorer
        dict(seed=4749, d=10, n=850, edge_prob=0.24, n_exo=3),  # held-out: largest
        dict(seed=4750, d=10, n=480, edge_prob=0.28, n_exo=3),  # held-out: largest + data-poorer
    ]
    out = []
    for p in specs:
        true_edges, forcing_cols, data = _gen_dag_sem(**p)
        out.append({"name": f"mesh{p['seed']}", "true_edges": true_edges,
                    "forcing_cols": forcing_cols, "data": data,
                    "d": p["d"], "n": p["n"]})
    return out


# ===================== structural Hamming distance =========================
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
            if t_any != p_any:                 # skeleton edge present in only one graph
                shd += 1
                continue
            if t_ij and p_ij and not t_ji and not p_ji:
                continue
            if t_ji and p_ji and not t_ij and not p_ij:
                continue
            shd += 1                            # both present but orientation mismatch/ambiguous
    return shd


# ===================== candidate answer handling ===========================
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
        E = len(true_edges)                    # SHD of the empty-graph baseline
        data = inst["data"]

        public = {
            "data": data.tolist(),
            "n_samples": int(inst["n"]),
            "n_nodes": int(d),
            "node_names": [_CHANNELS[k % len(_CHANNELS)] for k in range(d)],
            "forcing_nodes": [int(c) for c in inst["forcing_cols"]],
            "seed": int(20260474 + inst["n"] + d),
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
