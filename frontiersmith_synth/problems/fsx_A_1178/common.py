"""Shared deterministic construction for fsx_A_1178 (network-tomography-links).

Used by gen.py (emits the instance) and verify.py (regenerates ground truth
from the testId embedded in the input file). NOT importable by solutions --
the validation harness sandboxes solution runs so this module is invisible to
them; solutions must reconstruct any structural quantities (e.g. leaf counts)
themselves from the tree topology given on stdin.

Model: a rooted tree. Every edge e=(parent(v), v) has a hidden true
"log-loss" cost c_v >= 0. A handful of nodes are PROBED: for a probed node v
we are told the exact cumulative root-to-v sum D(v) = sum of c_u along the
root path (this is the path-matrix-forward measurement -- a real end-to-end
probe report). Because only some nodes are probed, a stretch of L>=2
unprobed edges lying between two consecutive probed checkpoints has its
SUM pinned exactly but its individual split left with (L-1) degrees of
freedom: this is the planted rank-deficiency. The topology is built so those
degrees of freedom never couple across siblings (every unprobed node has at
most one child that leads toward another probed node), so each ambiguous run
is an independent 1-equation-in-L-unknowns block -- tractable without a
general linear solve, but only exploitable by a solver that reasons about
which quantities the probes can and cannot separate.

True costs are generated as c_v = ALPHA * leaf_count(v) + jitter, i.e. they
correlate with how many leaves hang below the edge (a quantity fully visible
in the given topology, independent of the probe values) -- the
"tree-structure-prior" mechanism. A solver that ignores this and simply
splits an ambiguous run's known sum evenly across its edges (the obvious
first attempt) systematically mis-splits whenever the attached subtree sizes
along that run are skewed -- which the generator plants deliberately.
"""
import random

ALPHA = 2.0
JITTER = 0.35
MIN_COST = 0.05


def build_instance(test_id: int):
    rng = random.Random(900000 + test_id * 131 + 7)

    S = 4 + test_id
    if test_id >= 9:
        S += 1
    easy = test_id <= 2

    # ---- build the spine (root=0, nodes 1..S) ----
    parent = [-1]
    for i in range(1, S + 1):
        parent.append(i - 1)
    spine_nodes = list(range(0, S + 1))

    # ---- attach an orphan leaf bundle to every spine node, one deliberate
    #      "hotspot" bundle to plant skew for the ambiguous-run trap ----
    h = 1 + ((test_id * 7 + 3) % S)
    hotspot_size = 8 + test_id
    node_id = S
    for i in range(1, S + 1):
        k = hotspot_size if i == h else ((i + test_id) % 3)
        for _ in range(k):
            node_id += 1
            parent.append(i)
    N = node_id + 1

    # ---- leaf counts (single decreasing-id pass; parent[v] < v always) ----
    children = [[] for _ in range(N)]
    for v in range(1, N):
        children[parent[v]].append(v)
    leaf_count = [0] * N
    for v in range(N - 1, -1, -1):
        if not children[v]:
            leaf_count[v] = 1
        if v != 0:
            leaf_count[parent[v]] += leaf_count[v]

    # ---- true edge costs + cumulative D ----
    true_cost = [0.0] * N
    for v in range(1, N):
        jitter = rng.uniform(-JITTER, JITTER)
        true_cost[v] = max(MIN_COST, round(ALPHA * leaf_count[v] + jitter, 6))
    D = [0.0] * N
    for v in range(1, N):
        D[v] = round(D[parent[v]] + true_cost[v], 6)

    # ---- probe (measurement) set ----
    if easy:
        measured_nodes = set(spine_nodes[1:])
    elif test_id <= 4:
        measured_nodes = {spine_nodes[-1]}
    else:
        measured_nodes = {spine_nodes[S // 2], spine_nodes[-1]}
    measurements = [(v, D[v]) for v in sorted(measured_nodes)]

    return {
        "N": N, "parent": parent, "true_cost": true_cost,
        "leaf_count": leaf_count, "measurements": measurements,
        "S": S, "spine_nodes": spine_nodes,
    }


def compute_chains(N, parent, measured_dict):
    """Given the tree (parent[]) and a dict {node: given_cumulative_value}
    (root implicitly has value 0), partition every edge into either:
      - a "chain": an ordered (root-side -> leaf-side) list of edge ids
        (edge id == child node id) whose costs sum EXACTLY to a known
        target (an ambiguous run of L>=1 edges between two consecutive
        known checkpoints), or
      - "orphan": an edge touched by no measurement at all (target unknown).
    Returns (chains, orphan_edges) where chains is a list of
    (edge_list_top_to_bottom, target_sum).
    """
    chains = []
    support = set()
    for v in sorted(measured_dict.keys()):
        chain = []
        cur = v
        while True:
            chain.append(cur)
            nxt = parent[cur]
            if nxt == 0 or nxt in measured_dict:
                break
            cur = nxt
        chain.reverse()
        anchor = parent[chain[0]]
        anchor_val = 0.0 if anchor == 0 else measured_dict[anchor]
        target = measured_dict[v] - anchor_val
        chains.append((chain, target))
        support.update(chain)
    orphan_edges = [v for v in range(1, N) if v not in support]
    return chains, orphan_edges
