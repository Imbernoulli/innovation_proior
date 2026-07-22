#!/usr/bin/env python3
# Checker for fsx_S_0603 -- "A vein network that survives a cut".
# Deterministic scorer for a flow-network remodelling / fault-tolerance problem.
#
# The participant submits a LOADING SCHEDULE (weights over a menu of demand
# scenarios) and a reinforcement exponent gamma.  The checker runs the fixed
# "flux reinforces, else decays" adaptive-conductance rule to a stable network,
# then scores that network's transport quality under the WORST single-vein cut.
#
# Quality (lower cost = better) = worst-single-edge-removal of
#     [ aggregate transport dissipation  +  LAMBDA * sum of per-sink dissipations ].
# The aggregate term rewards efficient concentration; the per-sink-under-fault
# term rewards redundant loops that keep every sink fed after a cut.
import sys
import numpy as np

# ---------------- fixed, published mechanism constants ----------------
T_ITERS = 28          # remodelling iterations to the stable network
LAMBDA  = 0.25        # weight of per-sink resilience vs aggregate efficiency
POW     = 2.0         # score curvature (headroom shaping); trivial stays 0.1
CAP_D   = 1.0e6       # per-scenario dissipation cap (bounds a severed sink)
FLOOR   = 1.0e-9      # conductance floor for numerical positivity

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_tokens(path):
    with open(path) as f:
        return f.read().split()

# ---------- linear physics on the weighted resistor mesh ----------
def solve_multi(C, ii, jj, RHS, N, ref):
    """Solve L p = rhs for each column of RHS (N x m). Ground node ref.
    Returns potentials P (N x m). C = per-edge conductance (E,)."""
    L = np.zeros((N, N))
    np.add.at(L, (ii, ii), C)
    np.add.at(L, (jj, jj), C)
    np.add.at(L, (ii, jj), -C)
    np.add.at(L, (jj, ii), -C)
    keep = [n for n in range(N) if n != ref]
    Lr = L[np.ix_(keep, keep)]
    try:
        Pr = np.linalg.solve(Lr, RHS[keep, :])
    except np.linalg.LinAlgError:
        return None
    P = np.zeros((N, RHS.shape[1]))
    P[keep, :] = Pr
    return P

def dissip_cols(C, ii, jj, RHS, N, ref):
    """Vector of dissipations q^T p for each column of RHS (capped, finite)."""
    P = solve_multi(C, ii, jj, RHS, N, ref)
    if P is None:
        return np.full(RHS.shape[1], CAP_D)
    d = np.einsum('ij,ij->j', RHS, P)
    d = np.where(np.isfinite(d), d, CAP_D)
    d = np.clip(d, 0.0, CAP_D)
    return d

def remodel(weights, scen_RHS, gamma, ii, jj, E, N, ref):
    """Adaptive-conductance fixed point.  scen_RHS is (N x M); weights (M,).
    C_e <- (sum_m w_m * F_e(m)^2)^gamma, renormalised to mean 1."""
    C = np.ones(E)
    active = weights > 0
    if not active.any():
        return C
    W = weights[active]
    RHS = scen_RHS[:, active]
    for _ in range(T_ITERS):
        P = solve_multi(C, ii, jj, RHS, N, ref)
        if P is None:
            return None
        # edge flux for every active scenario:  F = C * (p_i - p_j)
        dP = P[ii, :] - P[jj, :]
        F = (C[:, None] * dP)
        Phi = (F * F) @ W                      # weighted mean squared flux
        Cn = np.power(Phi, gamma) + FLOOR
        s = Cn.sum()
        if not np.isfinite(s) or s <= 0:
            return None
        C = Cn * (E / s)
    return C

def worst_cut_cost(C, ii, jj, agg_RHS, sink_RHS, N, ref):
    """max over single-edge removals of  D_agg(faulted) + LAMBDA*sum D_sink(faulted)."""
    E = len(ii)
    RHS = np.concatenate([agg_RHS, sink_RHS], axis=1)   # col 0 = aggregate
    worst = 0.0
    for k in range(E):
        Ck = C.copy()
        Ck[k] = 0.0
        d = dissip_cols(Ck, ii, jj, RHS, N, ref)
        cost = d[0] + LAMBDA * d[1:].sum()
        if cost > worst:
            worst = cost
    return worst

def main():
    inp = read_tokens(sys.argv[1])
    out = read_tokens(sys.argv[2])

    # ---------------- parse instance ----------------
    try:
        it = iter(inp)
        R = int(next(it)); C = int(next(it))
        S = int(next(it))
        K = int(next(it))
        sinks = [int(next(it)) for _ in range(K)]
        gmin = float(next(it)); gmax = float(next(it))
        M = int(next(it))
        menu = []
        for _ in range(M):
            c = int(next(it))
            grp = [int(next(it)) for _ in range(c)]
            menu.append(grp)
    except Exception:
        fail("bad input")

    N = R * C
    # ---------------- build grid edges (4-neighbour mesh) ----------------
    edges = []
    idx = lambda r, cc: r * C + cc
    for r in range(R):
        for cc in range(C):
            if cc + 1 < C: edges.append((idx(r, cc), idx(r, cc + 1)))
            if r + 1 < R:  edges.append((idx(r + 1, cc), idx(r, cc)))
    E = len(edges)
    ii = np.array([e[0] for e in edges])
    jj = np.array([e[1] for e in edges])

    # scenario RHS columns for the menu
    scen_RHS = np.zeros((N, M))
    for m, grp in enumerate(menu):
        scen_RHS[S, m] = len(grp)
        for t in grp:
            scen_RHS[t, m] -= 1.0

    # aggregate (all sinks) and per-sink evaluation RHS
    agg_RHS = np.zeros((N, 1)); agg_RHS[S, 0] = K
    for t in sinks: agg_RHS[t, 0] -= 1.0
    sink_RHS = np.zeros((N, K))
    for a, t in enumerate(sinks):
        sink_RHS[S, a] = 1.0; sink_RHS[t, a] -= 1.0

    # ---------------- parse participant output ----------------
    try:
        gamma = float(out[0])
        w = np.array([float(x) for x in out[1:1 + M]], dtype=float)
    except Exception:
        fail("parse")
    if w.shape[0] != M:
        fail("need %d weights" % M)
    if not np.all(np.isfinite(w)) or not np.isfinite(gamma):
        fail("non-finite")
    if np.any(w < -1e-12):
        fail("negative weight")
    if gamma < gmin - 1e-9 or gamma > gmax + 1e-9:
        fail("gamma out of range")
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        fail("zero schedule")
    w = w / w.sum()

    # ---------------- remodel + score ----------------
    C_part = remodel(w, scen_RHS, gamma, ii, jj, E, N, ref=S)
    if C_part is None or not np.all(np.isfinite(C_part)):
        fail("remodel diverged")

    M_part = worst_cut_cost(C_part, ii, jj, agg_RHS, sink_RHS, N, ref=S)

    # internal baseline B: the do-nothing uniform mesh (participant gamma=0 reproduces it)
    C_uni = np.ones(E)
    B = worst_cut_cost(C_uni, ii, jj, agg_RHS, sink_RHS, N, ref=S)

    if M_part <= 1e-12:
        ratio = 1.0
    else:
        ratio = 0.1 * (B / M_part) ** POW
    ratio = max(0.0, min(1.0, ratio))
    print("agg+sink worst-cut cost=%.6f  baseline=%.6f  Ratio: %.6f" % (M_part, B, ratio))

if __name__ == "__main__":
    main()
