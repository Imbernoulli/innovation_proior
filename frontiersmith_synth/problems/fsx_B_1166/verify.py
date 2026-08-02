#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  (ans ignored)

Deterministic checker for the plume-source-inversion problem (format C).

Reads the testId back from <in> (line 1) and calls plumelib.build_instance(testId) to
regenerate the TRUE sparse source and the K_H held-out monitoring wells -- neither is
ever printed to the participant. The participant's artifact is an estimated release-rate
map over all M = N*N grid cells (row-major, index = i*N+j).

Feasibility (any violation -> "Ratio: 0.0"):
  * exactly M whitespace-separated numeric tokens, all finite
  * every rate >= -TOL (values in [-TOL,0) are clamped to 0.0)
  * total released mass <= 1.10 * B_mass (the stated budget, with float slack)

Objective (maximize), combining the three mechanisms:
  1. advection-diffusion-forward: the Green's-function forward operator (shared with
     gen.py) both builds the visible-well data the solver sees AND is used here to
     forward-simulate the predicted source map at the HELD-OUT wells.
  2. sparse-monitoring-wells: few visible wells relative to grid cells makes the visible
     linear system severely underdetermined/ill-conditioned -- many source maps explain
     the visible data equally well.
  3. source-sparsity-prior: recovery is scored by how much of the predicted MASS sits
     near the true (<=3) source cells (transportation/EMD-style localization), not by
     how well it merely fits the visible wells -- so a smeared multi-cell blob that fits
     the visible wells perfectly still scores poorly here.

  L (localization) = exp(-EMD(P, Q) / L0)     where
      P = predicted rate map, renormalized to a probability distribution over grid
          cells; Q = true rate map, renormalized similarly (support <= 3 cells).
      EMD = optimal-transport (Wasserstein-1) distance under Euclidean grid-cell cost,
          solved exactly via linear programming (scipy.optimize.linprog / HiGHS).
      L0  = 1.5 grid cells (localization length scale).
  H (held-out fit) = exp(-relerr),  relerr = ||C_pred - C_true||_2 / ||C_true||_2
      over the K_H held-out wells x MT observation times (never shown to the solver).
  F = 0.8*L + 0.2*H                    (localization-dominant; both terms present)

Baseline B: the checker's own trivial construction -- release the full mass budget
UNIFORMLY over every grid cell (maximally uninformative, "no idea where it is"). This
same construction is `solutions/trivial.py`, so it scores EXACTLY B by definition.

  sc = min(1000.0, 100.0*F/max(1e-9,B));  print("... Ratio: %.6f" % (sc/1000.0))

All arithmetic is deterministic (fixed testId-seeded RNG; the LP solver is deterministic
given fixed inputs); no wall-clock/GPU ever enters the score.
"""
import math
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plumelib import build_instance, cell_center, forward_conc, MT  # noqa: E402

TOL = 1e-6
W_L = 0.8
L0 = 1.5
BUDGET_SLACK = 1.10


def fail(reason):
    sys.stderr.write("reason: %s\n" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def emd(support_idx_mass, demand_pts, demand_mass, cells_enum, diag):
    """Exact Wasserstein-1 distance between the predicted mass distribution
    (support_idx_mass: list of (cell_index, prob_mass)) and the true source
    distribution (demand_pts/demand_mass, len<=3), Euclidean grid-cell cost."""
    if not support_idx_mass or not demand_pts:
        return diag
    ns = len(support_idx_mass)
    nd = len(demand_pts)
    supply_pts = [cell_center(*cells_enum[i]) for i, _m in support_idx_mass]
    P = np.array([m for _i, m in support_idx_mass], dtype=float)
    Q = np.array(demand_mass, dtype=float)
    if Q.sum() > 0:
        Q = Q * (P.sum() / Q.sum())  # neutralize float rounding so totals match exactly
    cost = np.zeros((ns, nd))
    for i, (x0, y0) in enumerate(supply_pts):
        for j, (x1, y1) in enumerate(demand_pts):
            cost[i, j] = math.hypot(x0 - x1, y0 - y1)
    c = cost.reshape(-1)
    A_eq = np.zeros((ns + nd, ns * nd))
    b_eq = np.zeros(ns + nd)
    for i in range(ns):
        A_eq[i, i * nd:(i + 1) * nd] = 1.0
        b_eq[i] = P[i]
    for j in range(nd):
        for i in range(ns):
            A_eq[ns + j, i * nd + j] = 1.0
        b_eq[ns + j] = Q[j]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return diag
    return float(res.fun)


def score_source_map(x, inst, diag):
    """Return F = W_L*L + (1-W_L)*H for a candidate rate map x (length M)."""
    N = inst["N"]
    cells_enum = inst["cells_enum"]
    Msum = sum(x)
    if Msum <= 1e-9:
        L = 0.0
    else:
        support = [(i, xc / Msum) for i, xc in enumerate(x) if xc > 1e-9]
        demand_pts = [cell_center(*c) for c in inst["true_cells"]]
        tot_true = sum(inst["true_rates"])
        demand_mass = [r / tot_true for r in inst["true_rates"]]
        d = emd(support, demand_pts, demand_mass, cells_enum, diag)
        L = math.exp(-d / L0)

    rates_true_full = [0.0] * inst["M"]
    idx_map = {c: k for k, c in enumerate(cells_enum)}
    for c, r in zip(inst["true_cells"], inst["true_rates"]):
        rates_true_full[idx_map[c]] = r

    sq_err = 0.0
    sq_true = 0.0
    for w in inst["held_wells"]:
        xw, yw = cell_center(*w)
        cpred = forward_conc(x, cells_enum, (xw, yw), inst["times"], inst["vx"], inst["vy"], inst["D"])
        ctrue = forward_conc(rates_true_full, cells_enum, (xw, yw), inst["times"], inst["vx"], inst["vy"], inst["D"])
        for a, b in zip(cpred, ctrue):
            sq_err += (a - b) ** 2
            sq_true += b * b
    relerr = math.sqrt(sq_err) / (math.sqrt(sq_true) + 1e-9)
    H = math.exp(-relerr)

    return W_L * L + (1.0 - W_L) * H


def main():
    try:
        in_toks = open(sys.argv[1]).read().split()
        test_id = int(in_toks[0])
        N_in = int(in_toks[1])
    except Exception:
        fail("bad instance file")

    inst = build_instance(test_id)
    if inst["N"] != N_in:
        fail("instance/testId mismatch")
    N = inst["N"]
    M = inst["M"]
    B_mass = inst["B_mass"]

    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output file")

    if len(out_toks) != M:
        fail("expected %d tokens, got %d" % (M, len(out_toks)))

    x = []
    for tk in out_toks:
        try:
            v = float(tk)
        except Exception:
            fail("non-numeric token %r" % tk)
        if not math.isfinite(v):
            fail("non-finite token %r" % tk)
        if v < -TOL:
            fail("negative rate %r" % tk)
        if v < 0.0:
            v = 0.0
        x.append(v)

    total = sum(x)
    if total > B_mass * BUDGET_SLACK + 1e-6:
        fail("total mass %.6f exceeds budget %.6f (x%.2f)" % (total, B_mass, BUDGET_SLACK))

    diag = math.sqrt(2.0) * N
    F = score_source_map(x, inst, diag)

    uniform = [B_mass / M] * M
    Bline = score_source_map(uniform, inst, diag)

    sc = min(1000.0, 100.0 * F / max(1e-9, Bline))
    sys.stderr.write("F=%.6f B=%.6f\n" % (F, Bline))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
