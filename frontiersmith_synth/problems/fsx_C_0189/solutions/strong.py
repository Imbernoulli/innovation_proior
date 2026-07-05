# TIER: strong
# Design the per-step schedule by simulating the exact update template internally and
# searching over (eta_k, gamma_k). Strategy:
#   1) probe a grid of constant scales around the stable step 1/L,
#   2) greedy coordinate descent, multiplicatively perturbing each step's eta/gamma.
# Deterministic (fixed traversal order, no wall-clock, no RNG). Beats constant EG by
# spending some steps aggressively (large eta) and others as fine polish.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
M = np.array(inst["M"], dtype=float)
q = np.array(inst["q"], dtype=float)
z0 = np.array(inst["z0"], dtype=float)
K = int(inst["K"])

L = float(np.linalg.norm(M, 2))
base = 1.0 / max(L, 1e-9)


def obj(eta, gamma):
    z = z0.copy()
    for k in range(K):
        g = M @ z + q
        if not np.all(np.isfinite(g)):
            return float("inf")
        w = z - gamma[k] * g
        z = z - eta[k] * (M @ w + q)
        if not np.all(np.isfinite(z)):
            return float("inf")
    r = M @ z + q
    if not np.all(np.isfinite(r)):
        return float("inf")
    return float(np.linalg.norm(r))


# 1) constant-schedule grid over (eta_scale, gamma_scale) * base
scales = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
best_eta = [base] * K
best_gamma = [base] * K
best = obj(best_eta, best_gamma)
for es in scales:
    for gs in scales:
        e = [es * base] * K
        g = [gs * base] * K
        v = obj(e, g)
        if v < best:
            best, best_eta, best_gamma = v, list(e), list(g)

# 2) coordinate descent on each step's parameters
factors = [0.5, 0.7, 0.85, 1.15, 1.4, 2.0]
eta = list(best_eta)
gamma = list(best_gamma)
for _pass in range(6):
    improved = False
    for k in range(K):
        for f in factors:
            cand_e = list(eta); cand_e[k] = eta[k] * f
            v = obj(cand_e, gamma)
            if v < best - 1e-14:
                best, eta = v, cand_e; improved = True
        for f in factors:
            cand_g = list(gamma); cand_g[k] = gamma[k] * f
            v = obj(eta, cand_g)
            if v < best - 1e-14:
                best, gamma = v, cand_g; improved = True
    if not improved:
        break

print(json.dumps({"eta": eta, "gamma": gamma}))
