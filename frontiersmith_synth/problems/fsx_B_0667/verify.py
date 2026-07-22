#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- deterministic checker for the relay-clock
cable-upgrade problem.

Feasibility: exactly M integer tokens (final weight per edge, in input
order), each >=1, budget sum(w_e-1) <= W, and per-node weighted-degree
sum(incident w_e) <= cap[node].  Any violation -> Ratio: 0.0.

Objective (maximize): F = algebraic connectivity (second-smallest
eigenvalue of the weighted graph Laplacian) of the submitted weighting.

Baseline B: the checker's own naive reference -- spend the FULL integer
budget via single-unit round-robin water-filling over a fixed
pseudo-random (but input-seeded, deterministic) edge order, respecting
the same per-node caps.  No spectral reasoning at all.

Ratio = min(1000, 100*F/B) / 1000  (matching B -> 0.1; 10x better caps at 1.0).
"""
import math
import random
import sys

import numpy as np


def fail(msg):
    print(f"Ratio: 0.0  # {msg}")
    sys.exit(0)


def algebraic_connectivity(n, edges, w):
    lap = np.zeros((n, n), dtype=np.float64)
    for (u, v), wt in zip(edges, w):
        lap[u, u] += wt
        lap[v, v] += wt
        lap[u, v] -= wt
        lap[v, u] -= wt
    vals = np.linalg.eigvalsh(lap)
    vals.sort()
    return float(vals[1])


def checker_baseline(n, m, w_budget, edges, cap, deg0):
    seed = (n * 1_000_003 + m * 9973 + w_budget * 31 + sum(cap) * 7) & 0x7FFFFFFF
    rng = random.Random(seed)
    order = list(range(m))
    rng.shuffle(order)

    bw = [1] * m
    remaining = [cap[i] - deg0[i] for i in range(n)]
    budget_left = w_budget
    max_rounds = w_budget + 2
    for _ in range(max_rounds):
        if budget_left <= 0:
            break
        progressed = False
        for e in order:
            if budget_left <= 0:
                break
            u, v = edges[e]
            if remaining[u] > 0 and remaining[v] > 0:
                bw[e] += 1
                remaining[u] -= 1
                remaining[v] -= 1
                budget_left -= 1
                progressed = True
        if not progressed:
            break
    return algebraic_connectivity(n, edges, bw)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    w_budget = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        edges.append((u, v))
    cap = [int(next(it)) for _ in range(n)]

    deg0 = [0] * n
    for (u, v) in edges:
        deg0[u] += 1
        deg0[v] += 1

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(out_toks) != m:
        fail(f"expected {m} tokens, got {len(out_toks)}")

    w = []
    for tok in out_toks:
        # strict integer literal (rejects "nan", "inf", "3.0", "1e9", etc.)
        if not (tok.lstrip("+-").isdigit()):
            fail(f"non-integer token {tok!r}")
        val = int(tok)
        if not math.isfinite(val):
            fail("non-finite value")
        if val < 1 or val > 1_000_000_000:
            fail(f"weight {val} out of range")
        w.append(val)

    extra = sum(x - 1 for x in w)
    if extra > w_budget:
        fail(f"budget exceeded: used {extra} > {w_budget}")

    deg_w = [0] * n
    for (u, v), wt in zip(edges, w):
        deg_w[u] += wt
        deg_w[v] += wt
    for i in range(n):
        if deg_w[i] > cap[i]:
            fail(f"node {i} weighted-degree {deg_w[i]} exceeds cap {cap[i]}")

    f_val = algebraic_connectivity(n, edges, w)
    if not math.isfinite(f_val) or f_val < -1e-7:
        fail("non-finite or negative objective")
    f_val = max(f_val, 0.0)

    b_val = checker_baseline(n, m, w_budget, edges, cap, deg0)
    b_val = max(b_val, 1e-9)

    sc = min(1000.0, 100.0 * f_val / b_val)
    print("F: %.6f B: %.6f Ratio: %.6f" % (f_val, b_val, sc / 1000.0))


if __name__ == "__main__":
    main()
