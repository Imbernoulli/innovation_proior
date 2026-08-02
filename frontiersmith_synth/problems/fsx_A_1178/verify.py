#!/usr/bin/env python3
"""Checker for fsx_A_1178 -- per-link loss (log-survival) rates from
root-anchored end-to-end probes on a tree.

Feasibility: the submitted per-edge costs must be finite, non-negative, and
must EXACTLY reproduce every given probe's cumulative sum (this is the
path-matrix-forward constraint -- any violation is disqualifying).

Objective: mean per-edge closeness to the hidden true costs (regenerated
deterministically from the testId embedded in the input, via common.py --
the same construction gen.py used). Edges inside an ambiguous run (the
sum is pinned by two probes but the individual split is not) can only be
guessed well by exploiting the tree-structure prior (leaf counts); edges with
no probe touching them at all are pure-prior guesses.

Internal baseline B: a deliberately naive but exactly-feasible construction
(dump each ambiguous run's whole sum onto its root-side edge, zero the rest;
guess a flat constant for untouched edges) -- this is what a trivial/do-
nothing reference implements, calibrating Ratio ~= 0.1 for it.
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_instance, compute_chains

TOL = 1e-3
ORPHAN_FLAT_GUESS = 1.0


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad checker invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    pos = 0
    test_id = int(toks[pos]); pos += 1
    N = int(toks[pos]); pos += 1
    M = int(toks[pos]); pos += 1
    parent = [-1] + [int(toks[pos + i]) for i in range(N - 1)]
    pos += N - 1
    measured_dict = {}
    for _ in range(M):
        v = int(toks[pos]); pos += 1
        dv = float(toks[pos]); pos += 1
        measured_dict[v] = dv

    inst = build_instance(test_id)
    if inst["N"] != N or inst["parent"] != parent:
        fail("input does not match the reconstructed instance (internal)")
    true_cost = inst["true_cost"]

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except FileNotFoundError:
        fail("no output produced")

    if len(out_toks) != N - 1:
        fail(f"expected {N - 1} numbers, got {len(out_toks)}")

    edge_cost = [0.0] * N
    for i, tok in enumerate(out_toks, start=1):
        try:
            x = float(tok)
        except ValueError:
            fail(f"token {i} ('{tok}') is not a number")
        if not math.isfinite(x):
            fail(f"token {i} is not finite")
        if x < -1e-6:
            fail(f"edge {i} cost {x} is negative")
        edge_cost[i] = max(0.0, x)

    D_pred = [0.0] * N
    for v in range(1, N):
        D_pred[v] = D_pred[parent[v]] + edge_cost[v]

    for v, dv in measured_dict.items():
        if abs(D_pred[v] - dv) > TOL:
            fail(f"probe at node {v}: predicted cumulative {D_pred[v]:.6f} "
                 f"!= measured {dv:.6f}")

    # Per-edge RELATIVE closeness (floored denominator): comparing against a
    # single global scale would let a handful of large-magnitude edges (the
    # ones near the root, which structurally always carry the most weight)
    # swamp the signal on the many small edges. Relative error keeps every
    # edge -- large or small -- equally informative about which strategy
    # actually identified the right split.
    def acc(pred, true):
        denom = max(true, 0.5)
        return max(0.0, 1.0 - abs(pred - true) / denom)

    F = sum(acc(edge_cost[v], true_cost[v]) for v in range(1, N)) / (N - 1)

    # ---- internal baseline: dump-on-first-edge + flat orphan guess ----
    chains, orphan_edges = compute_chains(N, parent, measured_dict)
    dumb_cost = [0.0] * N
    for chain, target in chains:
        dumb_cost[chain[-1]] = target
        for e in chain[:-1]:
            dumb_cost[e] = 0.0
    for e in orphan_edges:
        dumb_cost[e] = ORPHAN_FLAT_GUESS
    B = sum(acc(dumb_cost[v], true_cost[v]) for v in range(1, N)) / (N - 1)
    B = max(1e-6, B)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    print(f"F={F:.6f} B={B:.6f} Ratio: {ratio:.6f}")


if __name__ == "__main__":
    main()
