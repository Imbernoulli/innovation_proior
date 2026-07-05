# FROZEN evaluator -- do not ship to solvers as editable.
# Tide-pool ecology / convex-concave saddle: design an extragradient step-size &
# extrapolation schedule that drives the final equilibrium-gradient (residual) norm
# as low as possible after a FIXED iteration budget K. Deterministic, isolated.
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys, json, math, random
import numpy as np
import isorun

K_BUDGET = 4
N_INST = 10


def _rand_orth(rng, n):
    G = np.array([[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)], dtype=float)
    Q, R = np.linalg.qr(G)
    # fix signs deterministically so Q is a well-defined function of G
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return Q * d


def _spd(rng, n, mu, L):
    Q = _rand_orth(rng, n)
    if n == 1:
        eig = np.array([L])
    else:
        eig = np.linspace(mu, L, n)
    return (Q * eig) @ Q.T


def make_instances():
    """Deterministic, seeded family of convex-concave saddle operators.
    Saddle: min_x max_y  1/2 x^T A x + x^T B y - 1/2 y^T C y + a^T x - c^T y,  A,C >= 0.
    Monotone operator F(z) = M z + q,  z=(x,y),
        M = [[A, B], [-B^T, C]],  q = [a; c].
    We seek to minimize the equilibrium-gradient norm ||F(z_K)|| after K extragradient
    steps of a candidate-designed schedule. Later instances are stiffer (worse
    conditioning + stronger antisymmetric coupling) -> genuine held-out difficulty.
    """
    out = []
    for s in range(N_INST):
        rng = random.Random(1000 + s)
        n = 8 + s                        # per-side dim; total d = 2n in [16..34]
        L = 1.0
        mu = 0.5 * (0.40 ** s)           # strong-monotonicity shrinks -> stiffer
        A = _spd(rng, n, mu, L)
        C = _spd(rng, n, mu, L)
        beta = 1.0 + 0.40 * s            # antisymmetric coupling grows -> rotational
        B = beta * np.array([[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)], dtype=float)
        top = np.hstack([A, B])
        bot = np.hstack([-B.T, C])
        M = np.vstack([top, bot])
        q = np.array([rng.gauss(0.0, 1.0) for _ in range(2 * n)], dtype=float)
        z0 = np.array([rng.gauss(0.0, 1.0) for _ in range(2 * n)], dtype=float)
        out.append({
            "public": {
                "M": M.tolist(),
                "q": q.tolist(),
                "z0": z0.tolist(),
                "K": K_BUDGET,
            },
            "hidden": {},
        })
    return out


def _residual_norm(M, q, z):
    r = M @ z + q
    return float(np.linalg.norm(r))


def baseline(inst):
    """Trivial construction: take NO steps -> residual norm at the start point."""
    p = inst["public"]
    M = np.array(p["M"], dtype=float)
    q = np.array(p["q"], dtype=float)
    z0 = np.array(p["z0"], dtype=float)
    return _residual_norm(M, q, z0)


def _simulate(M, q, z0, eta, gamma, K):
    """Fixed extragradient template:
        g_k   = F(z_k)               = M z_k + q
        w_k   = z_k - gamma_k g_k
        z_{k+1} = z_k - eta_k F(w_k) = z_k - eta_k (M w_k + q)
    Returns final residual norm, or None if the trajectory left the finite range.
    """
    z = z0.astype(float).copy()
    for k in range(K):
        g = M @ z + q
        if not np.all(np.isfinite(g)):
            return None
        w = z - gamma[k] * g
        gw = M @ w + q
        z = z - eta[k] * gw
        if not np.all(np.isfinite(z)):
            return None
    r = M @ z + q
    if not np.all(np.isfinite(r)):
        return None
    return float(np.linalg.norm(r))


def score(inst, ans):
    """Strictly validate the candidate schedule, then recompute the objective."""
    p = inst["public"]
    K = p["K"]
    if not isinstance(ans, dict):
        return False, 0.0
    eta = ans.get("eta")
    gamma = ans.get("gamma")
    if not isinstance(eta, list) or not isinstance(gamma, list):
        return False, 0.0
    if len(eta) != K or len(gamma) != K:
        return False, 0.0
    try:
        eta = [float(x) for x in eta]
        gamma = [float(x) for x in gamma]
    except (TypeError, ValueError):
        return False, 0.0
    for v in eta + gamma:
        if not math.isfinite(v):
            return False, 0.0
        if abs(v) > 1e6:                 # reject absurd steps outright
            return False, 0.0
    M = np.array(p["M"], dtype=float)
    q = np.array(p["q"], dtype=float)
    z0 = np.array(p["z0"], dtype=float)
    obj = _simulate(M, q, z0, eta, gamma, K)
    if obj is None or not math.isfinite(obj):
        return False, 0.0
    return True, obj


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
