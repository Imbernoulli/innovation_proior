# TIER: strong
# Design a genuine SCHEDULE, not a constant.  Because the operator is affine, at each leg the
# next residual is g_{t+1} = g_t - a*(M g_t) - b*(M (g_t - g_{t-1})), which is LINEAR in (a, b).
# So we pick the per-leg gains that MINIMIZE ||g_{t+1}||^2 exactly, by solving the 2x2 normal
# equations over the two search directions u = M g_t and w = M (g_t - g_{t-1}).  This is a
# 2-dimensional Krylov residual-minimizing relay (Chebyshev/Orthomin-style acceleration): it
# adapts the step to the residual spectrum every leg and drives the operator norm far below any
# constant step -- yet, with the budget T well under the dimension and the ill-conditioned,
# strongly-coupled hard instances, it cannot reach the K=2.5-order target, leaving headroom.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
M = np.array(inst["M"], dtype=float)
q = np.array(inst["q"], dtype=float)
z0 = np.array(inst["z0"], dtype=float)
T = inst["T"]

z = z0.copy()
g = M @ z + q
gprev = g.copy()
A, B = [], []
for _ in range(T):
    u = M @ g
    w = M @ (g - gprev)
    G = np.array([[u @ u, u @ w], [u @ w, w @ w]])
    rhs = np.array([u @ g, w @ g])
    try:
        sol = np.linalg.solve(G + 1e-12 * np.eye(2), rhs)
        a_t, b_t = float(sol[0]), float(sol[1])
        if not (np.isfinite(a_t) and np.isfinite(b_t)):
            raise ValueError
    except Exception:
        denom = float(u @ u) + 1e-12
        a_t, b_t = float(u @ g) / denom, 0.0
    # keep within the evaluator's gain cap
    a_t = max(-1e4, min(1e4, a_t))
    b_t = max(-1e4, min(1e4, b_t))
    A.append(a_t)
    B.append(b_t)
    gp = g
    z = z - a_t * g - b_t * (g - gprev)
    gprev = gp
    g = M @ z + q

print(json.dumps({"a": A, "b": B}))
