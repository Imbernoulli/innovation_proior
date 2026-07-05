#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0381 -- "Rolling-Dough Robust Planning: Saddle-Point
Update Schedule Under a Round Budget"
(family: optimization-convergence-analysis; format B, eval_form: flops).

THEME.  A regional bakery cooperative plans daily production of n goods.  The
planner picks a production vector x (loaves/pastries per line); an adversarial
market / supplier picks a perturbation vector y (ingredient-price and demand
shocks) that pushes the cooperative's cost up.  The robust-planning equilibrium
is the saddle point of a convex-in-x, concave-in-y cost

    L(x, y) = 1/2 x^T A x + x^T B y - 1/2 y^T C y + b^T x - c^T y,

with A, C symmetric positive definite (so L is strongly convex in x and strongly
concave in y) and B the cross-coupling ("which shock hits which product line").
The unique equilibrium (x*, y*) satisfies the first-order (KKT) conditions
grad_x L = 0 and grad_y L = 0.  Stacking z = (x, y), these conditions read

    F(z) := [ grad_x L ; -grad_y L ] = M z + d = 0,
    M = [[A, B], [-B^T, C]],   d = [b; c].

M has positive-definite symmetric part diag(A, C), so F is a STRONGLY MONOTONE
operator -- but M is NON-symmetric (the coupling B makes its spectrum complex),
so the planning dynamics ROTATE and a naive gradient descent-ascent (GDA) crawls.

THE BUDGET (this is the "flops" currency).  The cooperative can only run a FIXED
number K of negotiation/planning ROUNDS; each round costs exactly ONE evaluation
of the operator F (one supply/demand response -- the matrix-vector product M z).
There is NO wall clock: cost = number of operator evaluations, hard-capped at K.
The planner therefore must DESIGN THE UPDATE RULE that squeezes the optimality
residual ||F(z_K)||_2 as small as possible within K rounds.

The evaluator runs a FIXED first-order template; the candidate supplies the
per-round scalar coefficients that drive it:

    z_{k+1} = z_k - alpha_k F(z_k) - beta_k F(z_{k-1}) + gamma_k (z_k - z_{k-1})
    for k = 0..K-1, with z_{-1} = z_0 and F(z_{-1}) = F(z_0).

This one-new-F-per-round template spans GDA (beta=gamma=0), optimistic GDA
(beta = -alpha), Polyak heavy-ball momentum (gamma>0), and their schedules.  It
CANNOT jump to the equilibrium: after K rounds the error z_K - z* is a degree-<=K
polynomial in M applied to z_0 - z*, so for K < dim the residual is bounded away
from 0 (a Krylov/Chebyshev limit).  Designing good schedules is the open problem.

OBJECTIVE (MINIMIZE, deterministic).  q_cand = ||F(z_K)||_2 recomputed by THIS
process from the candidate's schedule.  Per instance we also run the reference
NAIVE method (constant conservative GDA with step `ref_step`) to get q_base, and
normalize (minimization; naive baseline -> 0.1, better -> up to 1.0):

    r = min(1.0, 0.1 * q_base / max(q_cand, 1e-12)).

A schedule matching the naive baseline scores ~0.1; a worse (diverging) schedule
scores < 0.1 -> 0; the theoretically optimal degree-K method still leaves
headroom (r stays below 1.0 on every instance -- there is no easy optimum).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "block_size": n, "dim": 2n, "budget": K,
             "M": [[float]*dim]*dim, "d": [float]*dim, "z0": [float]*dim,
             "ref_step": float}
  stdout: ONE JSON object:
            {"alpha": [K floats], "beta": [K floats], "gamma": [K floats]}
          Each list must have exactly K finite floats with |value| <= 1e6.
          Wrong shape / non-finite / out-of-range / a crash / a timeout / a
          schedule that drives the iterate non-finite -> that instance scores 0.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS SANDBOX via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The reference
method and the residual re-computation happen in THIS parent process, so a
frame-walking / introspecting candidate learns nothing it was not already given.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def unif():                      # uniform in [0, 1)
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return unif


# ----------------------------- linear algebra (pure python) ----------------
def _matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _vnorm(v):
    return math.sqrt(sum(x * x for x in v))


def _rowinf(M):
    return max(sum(abs(x) for x in row) for row in M)


# ----------------------------- instance family -----------------------------
def _spd(u, n, mu, scale):
    """Symmetric positive-definite n x n matrix with min eigenvalue >= mu."""
    R = [[2.0 * u() - 1.0 for _ in range(n)] for _ in range(n)]
    A = [[(mu if i == j else 0.0) +
          scale * sum(R[k][i] * R[k][j] for k in range(n)) / n
          for j in range(n)] for i in range(n)]
    return A


def _build_instance(seed, n, mu, scale, kappa, K):
    """Deterministic saddle instance -> operator M (dim x dim), offset d, start z0."""
    u = _rng(seed)
    A = _spd(u, n, mu, scale)
    C = _spd(u, n, mu, scale)
    B = [[kappa * (2.0 * u() - 1.0) for _ in range(n)] for _ in range(n)]
    dim = 2 * n
    M = [[0.0] * dim for _ in range(dim)]
    for i in range(n):
        for j in range(n):
            M[i][j] = A[i][j]
            M[i][n + j] = B[i][j]
            M[n + i][j] = -B[j][i]
            M[n + i][n + j] = C[i][j]
    d = [2.0 * u() - 1.0 for _ in range(dim)]
    z0 = [0.0] * dim
    ref_step = 1.0 / _rowinf(M)          # conservative naive GDA step (safe, slow)
    return {"name": f"bakery{seed}", "block_size": n, "dim": dim,
            "budget": K, "M": M, "d": d, "z0": z0, "ref_step": ref_step}


def _build_instances():
    # (seed, n, mu, scale, kappa, K) -- vary conditioning (mu/scale) & rotation (kappa)
    specs = [
        (101, 6, 0.30, 3.0, 1.2, 6),
        (102, 6, 0.20, 4.0, 1.5, 6),
        (103, 5, 0.40, 3.0, 0.8, 6),
        (104, 6, 0.15, 4.0, 2.0, 6),
        (105, 7, 0.25, 3.5, 1.3, 6),
        (106, 6, 0.30, 3.0, 2.5, 6),
        # harder / larger held-out instances
        (207, 5, 0.50, 2.5, 1.0, 6),
        (208, 7, 0.20, 4.5, 1.8, 6),
        (209, 6, 0.35, 3.0, 1.1, 6),
        (210, 6, 0.18, 4.0, 2.2, 6),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- method template -----------------------------
def _run_schedule(M, d, z0, alpha, beta, gamma, K):
    """Run the fixed first-order template. Return final iterate z_K (or None if
    it goes non-finite)."""
    z_prev = list(z0)
    z = list(z0)
    F_prev = [_matvec(M, z0)[i] + d[i] for i in range(len(z0))]
    for k in range(K):
        Fz = [_matvec(M, z)[i] + d[i] for i in range(len(z))]
        znew = [z[i] - alpha[k] * Fz[i] - beta[k] * F_prev[i]
                + gamma[k] * (z[i] - z_prev[i]) for i in range(len(z))]
        for val in znew:
            if val != val or val in (float("inf"), float("-inf")):
                return None
        z_prev = z
        z = znew
        F_prev = Fz
    return z


def _baseline_residual(inst):
    """Naive reference: constant conservative GDA with step ref_step."""
    K = inst["budget"]
    eta = inst["ref_step"]
    z = _run_schedule(inst["M"], inst["d"], inst["z0"],
                      [eta] * K, [0.0] * K, [0.0] * K, K)
    Fz = [_matvec(inst["M"], z)[i] + inst["d"][i] for i in range(len(z))]
    return _vnorm(Fz)


# ----------------------------- validation ----------------------------------
def _residual_of_answer(inst, answer):
    """Validate the schedule and recompute ||F(z_K)||. Return float or None."""
    if not isinstance(answer, dict):
        return None
    K = inst["budget"]
    sched = []
    for key in ("alpha", "beta", "gamma"):
        arr = answer.get(key)
        if not isinstance(arr, list) or len(arr) != K:
            return None
        vals = []
        for x in arr:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None
            xf = float(x)
            if xf != xf or xf in (float("inf"), float("-inf")):
                return None
            if abs(xf) > 1e6:
                return None
            vals.append(xf)
        sched.append(vals)
    alpha, beta, gamma = sched
    z = _run_schedule(inst["M"], inst["d"], inst["z0"], alpha, beta, gamma, K)
    if z is None:
        return None
    Fz = [_matvec(inst["M"], z)[i] + inst["d"][i] for i in range(len(z))]
    q = _vnorm(Fz)
    if q != q or q in (float("inf"), float("-inf")):
        return None
    return q


def _public_view(inst):
    return {"name": inst["name"], "block_size": inst["block_size"],
            "dim": inst["dim"], "budget": inst["budget"],
            "M": [list(row) for row in inst["M"]], "d": list(inst["d"]),
            "z0": list(inst["z0"]), "ref_step": inst["ref_step"]}


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = _baseline_residual(inst)
        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _residual_of_answer(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 * q_base / max(q_cand, 1e-12)
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
